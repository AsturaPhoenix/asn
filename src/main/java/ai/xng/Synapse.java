package ai.xng;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;
import java.util.Map.Entry;
import java.util.WeakHashMap;
import java.util.concurrent.TimeUnit;
import io.reactivex.Observable;
import io.reactivex.disposables.Disposable;
import io.reactivex.subjects.PublishSubject;
import io.reactivex.subjects.ReplaySubject;
import io.reactivex.subjects.Subject;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;
import lombok.val;

/**
 * Represents the incoming logical junction of input node signals towards a
 * specific output node.
 * 
 * Minimal effort is made to preserve activation across serialization boundaries
 * since behavior while the system is down is discontinuous anyway. In the
 * future, it is likely that either we will switch to relative time and fully
 * support serialization or else completely clear activation on deserialization.
 */
public class Synapse implements Serializable {
  private static final long serialVersionUID = 1779165354354490167L;

  public static final long DEBOUNCE_PERIOD = 2;
  private static final float DECAY_MARGIN = .2f;
  private static final float THRESHOLD = 1;

  public class ContextualState {
    private final Subject<Long> rxEvaluate;
    private final Map<Profile, Evaluation> evaluations = Collections.synchronizedMap(new WeakHashMap<>());
    Evaluation lastActivation;

    public ContextualState(final Context context) {
      rxEvaluate = PublishSubject.create();
      rxEvaluate.switchMap(t -> evaluate(context, t).doFinally(context::releaseRef)).subscribe(t -> {
        rxOutput.onNext(new Node.Activation(context, t));
      });
    }

    public void reinforce(Optional<Long> time, Optional<Long> decayPeriod, float weight) {
      synchronized (evaluations) {
        for (final Entry<Profile, Evaluation> evaluation : evaluations.entrySet()) {
          final long t = evaluation.getValue().time;
          float wt = weight;
          if (time.isPresent()) {
            if (t > time.get())
              continue;
            if (decayPeriod.isPresent()) {
              long t0 = time.get() - decayPeriod.get();
              if (t <= t0)
                continue;
              wt *= (float) (t - t0) / decayPeriod.get();
            }
          }

          evaluation.getKey().coefficient.add(evaluation.getValue().value, wt);
        }
      }
    }
  }

  public class Profile {
    @Getter
    private Distribution coefficient;
    @Getter
    private long decayPeriod; // linear for now
    private final Node incoming;
    private final Disposable subscription;

    private Profile(final Node incoming) {
      this.coefficient = new ThresholdDistribution(0);
      this.incoming = incoming;
      resetDecay();
      subscription = incoming.rxActivate().subscribe(this::onActivate);
    }

    public void resetDecay() {
      // The default decay should be roughly proportional to the
      // refractory period of the source node as nodes with shorter
      // refractory periods are likely to be evoked more often, possibly
      // spuriously, and should thus get out of the way faster. By the
      // time the refractory period has elapsed and the node may thus be
      // activated again, we want this activation to be decayed by at the
      // decay margin.
      decayPeriod = Math.max((long) (incoming.getRefractory() / DECAY_MARGIN), 1);
    }

    private void onActivate(final Node.Activation activation) {
      // Schedules an evaluation in the appropriate context, which will
      // sum the incoming signals.
      activation.context.addRef();
      activation.context.synapseState(Synapse.this).rxEvaluate.onNext(activation.timestamp);
    }

    public float getValue(final Context context, final long time) {
      // We lazily re-evaluate inhibitory coefficients, so check whether
      // we need to re-evaluate now.
      long lastActivation = incoming.getLastActivation(context);
      long dt = Math.max(time - lastActivation, 0);

      if (dt >= decayPeriod) {
        return 0;
      }

      final ContextualState contextualState = context.synapseState(Synapse.this);
      // TODO(rosswang): it may be an interesting simplification to
      // restrict that nodes may only be activated once per context. Then
      // we might be able to get rid of refractory periods and decays, but
      // it would have implications for temporal processing (in particular
      // we'd force discrete time steps).
      final Evaluation lastEvaluation = contextualState.evaluations.get(this);
      final float v0;
      if (lastEvaluation == null || lastEvaluation.time != lastActivation) {
        v0 = coefficient.generate();
        contextualState.evaluations.put(this, new Evaluation(lastActivation, v0));
      } else {
        v0 = lastEvaluation.value;
      }
      return v0 * (1 - dt / (float) decayPeriod);
    }

    public float getLastCoefficient(final Context context) {
      return context.synapseState(Synapse.this).evaluations.get(this).value;
    }

    private long getZero(final Context context) {
      return incoming.getLastActivation(context) + decayPeriod;
    }
  }

  @Getter
  private transient Map<Node, Profile> inputs;

  private transient Subject<Node.Activation> rxOutput;
  private transient Subject<ContextualEvaluation> rxValue;

  public Synapse() {
    init();
  }

  private void init() {
    inputs = Collections.synchronizedMap(new WeakHashMap<>());
    rxOutput = PublishSubject.create();
    rxValue = ReplaySubject.createWithSize(EVALUATION_HISTORY);
  }

  @RequiredArgsConstructor
  @ToString
  @EqualsAndHashCode
  public static class Evaluation {
    public final long time;
    public final float value;
  }

  @RequiredArgsConstructor
  @ToString
  @EqualsAndHashCode
  public static class ContextualEvaluation {
    public final Context context;
    public final Evaluation evaluation;
  }

  public static final int EVALUATION_HISTORY = 10;

  public Observable<ContextualEvaluation> rxValue() {
    return rxValue;
  }

  /**
   * Emits an activation signal or schedules a re-evaluation at a future time,
   * depending on current state.
   */
  private Observable<Long> evaluate(final Context context, final long time) {
    val synapseState = context.synapseState(this);
    // Synchronize for consistency during reinforcement.
    synchronized (synapseState.evaluations) {
      final float value = getValue(context, time);
      val evaluation = new Evaluation(time, value);
      rxValue.onNext(new ContextualEvaluation(context, evaluation));

      if (value >= THRESHOLD) {
        synapseState.lastActivation = evaluation;
        return Observable.just(time);
      } else {
        final long nextCrit = getNextCriticalPoint(context, time);
        if (nextCrit == Long.MAX_VALUE) {
          return Observable.empty();
        } else {
          return Observable.timer(nextCrit - time, TimeUnit.MILLISECONDS).flatMap(x -> evaluate(context, nextCrit));
        }
      }
    }
  }

  public float getValue(final Context context, final long time) {
    float value = 0;
    synchronized (inputs) {
      for (final Profile profile : inputs.values()) {
        value += profile.getValue(context, time);
      }
    }
    return value;
  }

  /**
   * Gets the next time the synapse should be evaluated if current conditions
   * hold. This is the minimum of the next time the synapse would cross the
   * activation threshold given current conditions, and the zeros of the
   * activations involved. Activations that have already fully decayed do not
   * affect this calculation.
   * 
   * This is a very conservative definition and can be optimized further.
   */
  private long getNextCriticalPoint(final Context context, final long time) {
    float totalValue = 0, totalDecayRate = 0;
    long nextZero = Long.MAX_VALUE;
    boolean hasInhibitory = false;
    synchronized (inputs) {
      for (final Profile profile : inputs.values()) {
        final float value = profile.getValue(context, time);
        if (value != 0) {
          totalValue += value;
          float coefficient = profile.getLastCoefficient(context);
          if (coefficient < 0) {
            hasInhibitory = true;
          }
          totalDecayRate += coefficient / profile.decayPeriod;
          nextZero = Math.min(nextZero, profile.getZero(context));
        }
      }
    }

    // This is a simple optimization for a common case.
    if (!hasInhibitory) {
      return Long.MAX_VALUE;
    }

    final long untilThresh = (long) ((1 - totalValue) / -totalDecayRate);
    return untilThresh <= 0 ? nextZero : Math.min(untilThresh + time, nextZero);
  }

  public Observable<Node.Activation> rxActivate() {
    return rxOutput;
  }

  private void writeObject(final ObjectOutputStream o) throws IOException {
    o.defaultWriteObject();
    synchronized (inputs) {
      o.writeInt(inputs.size());
      for (final Entry<Node, Profile> entry : inputs.entrySet()) {
        o.writeObject(entry.getKey());
        o.writeObject(entry.getValue().coefficient);
        o.writeLong(entry.getValue().decayPeriod);
      }
    }
  }

  private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
    o.defaultReadObject();
    init();
    final int size = o.readInt();
    for (int i = 0; i < size; i++) {
      final Node node = (Node) o.readObject();
      final Profile profile = new Profile(node);
      profile.coefficient = (Distribution) o.readObject();
      profile.decayPeriod = o.readLong();
      inputs.put(node, profile);
    }
  }

  public Profile profile(final Node node) {
    return inputs.computeIfAbsent(node, Profile::new);
  }

  public Synapse setCoefficient(final Node node, final float coefficient) {
    val profile = profile(node);
    profile.coefficient.set(coefficient);
    return this;
  }

  public float getCoefficient(final Node node) {
    final Profile profile = inputs.get(node);
    return profile == null ? 0 : profile.coefficient.getMode();
  }

  /**
   * @param node      the input node
   * @param decayRate the linear signal decay period, in milliseconds from
   *                  activation to 0
   */
  public Synapse setDecayPeriod(final Node node, final long decayPeriod) {
    profile(node).decayPeriod = decayPeriod;
    return this;
  }

  public long getDecayPeriod(final Node node) {
    final Profile profile = inputs.get(node);
    return profile == null ? 0 : profile.decayPeriod;
  }

  public void dissociate(final Node node) {
    final Profile profile = inputs.remove(node);
    if (profile != null) {
      profile.subscription.dispose();
    }
  }

  public Evaluation getLastEvaluation(final Context context, final Node incoming) {
    return context.synapseState(this).evaluations.get(inputs.get(incoming));
  }
}
