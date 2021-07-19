package ai.xng.constructs;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.lang.ref.WeakReference;
import java.util.function.Consumer;

import ai.xng.ActionCluster;
import ai.xng.IntegrationProfile;
import ai.xng.Posterior;
import ai.xng.PosteriorCluster;
import ai.xng.Scheduler;
import io.reactivex.Observer;
import io.reactivex.disposables.Disposable;
import lombok.val;

/**
 * Note that although this class is serializable in terms of wiring, since
 * integrators are transient, so is activation state.
 */
public abstract class CoincidentEffect<T extends Posterior> implements Serializable {
  public static class Lambda<T extends Posterior> extends CoincidentEffect<T> {
    public interface Effect<T extends Posterior> extends Consumer<T>, Serializable {
    }

    private final Effect<T> effect;

    public Lambda(final ActionCluster actionCluster, final PosteriorCluster<? extends T> effectCluster,
        final Effect<T> effect) {
      super(actionCluster, effectCluster);
      this.effect = effect;
    }

    @Override
    protected void apply(final T node) {
      effect.accept(node);
    }
  }

  public static class Curry<T extends Posterior> extends CoincidentEffect<T> {
    private T valueNode;

    public T require() {
      if (!hasValueNode()) {
        throw new IllegalStateException("Required value has not been curried.");
      }
      val valueNode = this.valueNode;
      reset();
      return valueNode;
    }

    public boolean hasValueNode() {
      return valueNode != null;
    }

    public Curry(final ActionCluster actionCluster, final PosteriorCluster<? extends T> effectCluster) {
      super(actionCluster, effectCluster);
    }

    @Override
    protected void apply(final T node) {
      if (!hasValueNode()) {
        valueNode = node;
        // TODO: Once timing is more robust, it may be more useful to throw an exception
        // if a value has already been curried.
      }
    }

    public void reset() {
      valueNode = null;
    }
  }

  public final ActionCluster.Node node;
  public final PosteriorCluster<? extends T> cluster;

  public CoincidentEffect(final ActionCluster actionCluster, final PosteriorCluster<? extends T> effectCluster) {
    node = actionCluster.new Node(this::onActivate);
    cluster = effectCluster;
    subscribe();
  }

  protected abstract void apply(T node);

  /**
   * Subclass implementations must call {@code super.onActivate()}. For a given
   * activation, this method is guaranteed to be called before any coincident
   * {@link #apply(Posterior)} calls.
   */
  protected void onActivate() {
    for (val recent : cluster.activations()) {
      if (recent.getIntegrator().isPending()) {
        // This case will be handled by the subscription.
        continue;
      }
      if (recent.getLastActivation().get() < Scheduler.global.now() - IntegrationProfile.PERSISTENT.period()) {
        // This assumes that PERSISTENT is an upper bound on integration curve periods.
        break;
      }

      if (recent.getIntegrator().isActive()) {
        apply(recent);
      }
    }
  }

  /**
   * Avoids a lapsed-listener leak using a weak reference to the outer class.
   * There is no test covering this leak since it manifests only in transient
   * memory, the usage of which we do not have a great way to guage.
   * <p>
   * TODO: We might be able to do test this with Instrumentation.
   */
  private static class WeakSubscription<T extends Posterior> implements Observer<T> {
    final WeakReference<CoincidentEffect<T>> weakParent;
    Disposable subscription;

    WeakSubscription(final CoincidentEffect<T> parent) {
      weakParent = new WeakReference<>(parent);
    }

    @Override
    public void onSubscribe(final Disposable d) {
      subscription = d;
    }

    public void onNext(final T node) {
      val parent = weakParent.get();
      if (parent == null) {
        subscription.dispose();
      } else {
        val integrator = parent.node.getIntegrator();
        if (!integrator.isPending() && integrator.isActive()) {
          parent.apply(node);
        }
      }
    }

    @Override
    public void onComplete() {
    }

    @Override
    public void onError(Throwable e) {
    }
  }

  private void subscribe() {
    cluster.rxActivations().subscribe(new WeakSubscription<T>(this));
  }

  private void readObject(final ObjectInputStream o) throws ClassNotFoundException, IOException {
    o.defaultReadObject();
    subscribe();
  }
}
