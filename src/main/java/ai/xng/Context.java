package ai.xng;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.function.Supplier;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.MoreExecutors;

import io.reactivex.Observable;
import io.reactivex.disposables.CompositeDisposable;
import io.reactivex.functions.Consumer;
import io.reactivex.subjects.BehaviorSubject;
import io.reactivex.subjects.Subject;
import lombok.Getter;
import lombok.val;

/**
 * A node value that signifies that a node is behaving as an
 * evaluation/execution context. Properties of such nodes are used as contextual
 * memory.
 * 
 * Although this class is {@link Serializable} for use as a node value, most of
 * its state is transient. That is, although contexts are nodes, activations are
 * not expected to remain valid across serialization.
 */
public class Context implements Serializable {
  public interface SerializableSupplier<T> extends Supplier<T>, Serializable {
  }

  private static final long serialVersionUID = 3575759895941990217L;

  private transient Map<Node, Node.ContextualState> nodeStates;
  private transient Map<Synapse, Synapse.ContextualState> synapseStates;

  private final RecencyQueue<Node> activations = new RecencyQueue<>();

  private transient Subject<Boolean> rxActive;
  private transient int refCount;
  private transient Object refMutex;

  private transient CompositeDisposable contextDisposable;
  private transient DisposingWeakReference<Context> weakThis;

  private final SerializableSupplier<Executor> executorSupplier;
  @Getter
  private transient ContextScheduler scheduler;

  // The node representing this context.
  public final Node node;

  /**
   * The exception handler for this context. This defaults to completing
   * {@link #continuation} exceptionally, unless overridden.
   */
  public transient Consumer<Throwable> exceptionHandler;

  /**
   * Sets {@link #exceptionHandler} to publish exceptions to {@link #rxActive()}.
   * This is useful in tests.
   */
  public void fatalOnExceptions() {
    exceptionHandler = rxActive::onError;
  }

  // A synchronous bridge for this context, completed when the return value is
  // first set or the context first becomes idle, or completed exceptionally on
  // unhandled exception.
  private transient CompletableFuture<Void> continuation;

  public CompletableFuture<Void> continuation() {
    return continuation;
  }

  public Context(final Function<? super Context, Node> nodeFactory,
      final SerializableSupplier<Executor> executorSupplier) {
    this.executorSupplier = executorSupplier;
    init();
    node = nodeFactory.apply(this);
  }

  /**
   * Creates a self-sufficient context with a dedicated thread, suitable for
   * tests.
   */
  public static Context newDedicated() {
    return new Context(Node::new, Executors::newSingleThreadExecutor);
  }

  /**
   * Creates a self-sufficient context using a direct scheduler, suitable for
   * tests.
   */
  public static Context newImmediate() {
    return new Context(Node::new, MoreExecutors::directExecutor);
  }

  private void init() {
    nodeStates = new HashMap<>();
    synapseStates = new HashMap<>();

    refMutex = new Object();
    rxActive = BehaviorSubject.createDefault(false);

    contextDisposable = new CompositeDisposable();
    weakThis = new DisposingWeakReference<>(this, contextDisposable);

    scheduler = new ContextScheduler(executorSupplier.get());
    scheduler.start();

    continuation = new CompletableFuture<>();
    exceptionHandler = continuation::completeExceptionally;
  }

  private void writeObject(final ObjectOutputStream stream) throws IOException {
    scheduler.pause();
    try {
      stream.defaultWriteObject();
    } finally {
      scheduler.resume();
    }
  }

  private void readObject(final ObjectInputStream stream) throws ClassNotFoundException, IOException {
    stream.defaultReadObject();
    init();
  }

  public Observable<Boolean> rxActive() {
    return rxActive;
  }

  public class Ref implements AutoCloseable {
    private AtomicBoolean active = new AtomicBoolean(true);

    public Ref() {
      synchronized (refMutex) {
        // In the past, we used an AtomicInteger instead of a mutex, but that resulted
        // in nonsense ordering in rxActive (eg. true true false false).
        if (refCount++ == 0) {
          rxActive.onNext(true);
        }
      }
    }

    @Override
    public void close() {
      if (!active.getAndSet(false)) {
        System.err.println("Warning: context ref already closed.");
        Thread.dumpStack();
        return;
      }

      synchronized (refMutex) {
        final int refs = --refCount;
        assert refs >= 0;
        if (refs == 0) {
          rxActive.onNext(false);
          continuation.complete(null);
        }
      }
    }
  }

  public void blockUntilIdle() {
    rxActive.filter(active -> !active).blockingFirst();
  }

  public void blockUntilIdle(final Duration timeout) {
    rxActive.filter(active -> !active).timeout(timeout.toMillis(), TimeUnit.MILLISECONDS).blockingFirst();
  }

  /**
   * Gets the contextual state for the given node, creating if absent. Must be
   * called from the scheduler thread.
   */
  public Node.ContextualState nodeState(final Node node) {
    assert scheduler.isOnThread();

    return nodeStates.computeIfAbsent(node, n -> {
      val link = activations.new Link(n);
      // local to avoid capturing this (lapsed listener)
      val weakThis = this.weakThis;
      val subscription = n.rxActivate().filter(a -> a.context == weakThis.get()).subscribe(a -> link.promote());
      contextDisposable.add(subscription);

      return n.new ContextualState(this);
    });
  }

  /**
   * Gets the contextual state for the given synapse, creating if absent. Must be
   * called from the scheduler thread.
   */
  public Synapse.ContextualState synapseState(final Synapse synapse) {
    assert scheduler.isOnThread();

    return synapseStates.computeIfAbsent(synapse, s -> s.new ContextualState(this));
  }

  public Node require(final Node property) {
    final Node value = node.properties.get(property);
    if (value == null) {
      throw new ContextException(this, property);
    }
    return value;
  }

  /**
   * Applies reinforcement to the context. This may be called from any thread but
   * executes on the scheduler thread.
   */
  public void reinforce(final Optional<Long> time, final Optional<Long> decayPeriod, final float weight) {
    scheduler.ensureOnThread(() -> {
      for (final HebbianReinforcementWindow window = new HebbianReinforcementWindow(time); window.hasNext();) {
        hebbianReinforcement(window.next(), time, weight * HEBBIAN_EXPLICIT_WEIGHT_FACTOR);
      }

      // There are definitely more efficient ways to do this.
      for (final Synapse.ContextualState synapseState : synapseStates.values()) {
        synapseState.reinforce(time, decayPeriod, weight);
      }
    });
  }

  public void hebbianReinforcement(final Node posterior, final float weight) {
    scheduler.ensureOnThread(() -> {
      for (final HebbianReinforcementWindow window = new HebbianReinforcementWindow(Optional.empty()); window
          .hasNext();) {
        final List<Node> snapshot = window.next();
        if (snapshot.get(0) == posterior) {
          hebbianReinforcement(snapshot, Optional.empty(), weight);
          return;
        }
      }
    });
  }

  public static long HEBBIAN_MAX_GLOBAL = 15000, HEBBIAN_MAX_LOCAL = 500;
  public static float HEBBIAN_IMPLICIT_WEIGHT = .1f, HEBBIAN_EXPLICIT_WEIGHT_FACTOR = .1f;

  private class HebbianReinforcementWindow implements Iterator<List<Node>> {
    final Optional<Long> time;
    final Iterator<Node> activations = Context.this.activations.iterator();
    Node next;
    final Queue<Node> deck = new ArrayDeque<>();

    HebbianReinforcementWindow(final Optional<Long> time) {
      this.time = time;
      do {
        draw();
        // Drain until we reach the effective time.
      } while (next != null && time.isPresent() && next.getLastActivation(Context.this) > time.get());
    }

    private void draw() {
      assert scheduler.isOnThread();

      if (activations.hasNext()) {
        next = activations.next();
        final long lastActivation = next.getLastActivation(Context.this);
        if (time.isPresent() && lastActivation < time.get() - HEBBIAN_MAX_GLOBAL || lastActivation == 0) {
          next = null;
        }
      } else {
        next = null;
      }
    }

    @Override
    public boolean hasNext() {
      return next != null || !deck.isEmpty();
    }

    @Override
    public List<Node> next() {
      while (next != null && (deck.isEmpty()
          || next.getLastActivation(Context.this) >= deck.peek().getLastActivation(Context.this) - HEBBIAN_MAX_LOCAL)) {
        deck.add(next);
        draw();
      }

      final List<Node> snapshot = ImmutableList.copyOf(deck);

      deck.remove();

      return snapshot;
    }
  }

  private void hebbianReinforcement(final List<Node> snapshot, final Optional<Long> time, final float baseWeight) {
    final Node posterior = snapshot.get(0);
    if (!(posterior instanceof SynapticNode synapticPosterior))
      return;

    final Synapse synapse = synapticPosterior.synapse;
    if (synapse == null)
      return;

    final long posteriorTime = posterior.getLastActivation(this);
    final float posteriorWeight = time.isPresent()
        ? baseWeight * (1 - (float) (time.get() - posteriorTime) / HEBBIAN_MAX_GLOBAL)
        : baseWeight;

    final int nPriors = snapshot.size() - 1;

    for (final Node prior : snapshot.subList(1, snapshot.size())) {
      Synapse.Evaluation priorEvaluation = synapse.getPrecedingEvaluation(this, prior, posteriorTime);
      assert priorEvaluation == null || priorEvaluation.time > 0;

      if (priorEvaluation == null) {
        // New connection; treat as a prior evaluation of 0.
        priorEvaluation = new Synapse.Evaluation(prior.getLastActivation(this), 0);
        // TODO: Also add initial inertia based on node activation history. For now this
        // is just 1 at 0.
      }

      final float weight = posteriorWeight * (1 - (float) (posteriorTime - priorEvaluation.time) / HEBBIAN_MAX_LOCAL);
      final Distribution distribution = synapse.profile(prior).getCoefficient();
      float margin = Synapse.THRESHOLD - synapse.getValue(this, priorEvaluation.time);
      if (margin <= 0) {
        margin = 0;
      } else {
        margin /= nPriors;
      }
      final float target = Math.min(margin + distribution.getMode(), 1);
      distribution.add(target, weight);
    }
  }

  /**
   * Captures a snapshot of nodes by activation recency. This should be called
   * from the dispatch thread.
   */
  public ImmutableList<Node> snapshotNodes() {
    return ImmutableList.copyOf(activations);
  }
}
