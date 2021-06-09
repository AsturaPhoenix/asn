package ai.xng;

import java.io.IOException;
import java.io.ObjectInputStream;

import ai.xng.util.EvictingIterator;
import lombok.Getter;
import lombok.val;

public abstract class PosteriorCluster<T extends Posterior> extends Cluster<T> {
  /**
   * Plasticity is disabled for now until requirements become clearer. It's likely
   * that the STDP behavior needs to be adjusted to penalize connections that do
   * not trigger. However, it is also possible that explicit plasticity in the
   * generalized capture mechanism is preferable to "always-on" plasticity in this
   * manner.
   */
  public static final float DEFAULT_PLASTICITY = 0;

  protected class ClusterNodeTrait extends Cluster<T>.ClusterNodeTrait {
    private final T owner;
    private final WeakSerializableRecencyQueue<T>.Link priorTouchLink;

    @Getter
    private transient ThresholdIntegrator integrator;

    public ClusterNodeTrait(final T owner) {
      super(owner);
      this.owner = owner;
      priorTouchLink = priorTouches.new Link(owner);
      init();
    }

    private void init() {
      integrator = new ThresholdIntegrator() {
        @Override
        protected void onThreshold() {
          owner.activate();
        }

        @Override
        public Spike add(IntegrationProfile profile, float magnitude) {
          val spike = super.add(profile, magnitude);
          priorTouchLink.promote();
          return spike;
        }
      };
    }

    private void readObject(final ObjectInputStream stream) throws IOException, ClassNotFoundException {
      stream.defaultReadObject();
      init();
    }
  }

  private final WeakSerializableRecencyQueue<T> priorTouches = new WeakSerializableRecencyQueue<>();

  /**
   * An iterable view of node by prior activation recency. Nodes are evicted from
   * this view when their integrators become empty.
   */
  public Iterable<T> priorTouches() {
    return () -> new EvictingIterator<>(priorTouches.iterator()) {
      @Override
      protected boolean shouldEvict(final T item) {
        item.getIntegrator().evict();
        return item.getIntegrator().isEmpty();
      }
    };
  }

  @Getter
  private float plasticity = DEFAULT_PLASTICITY;

  public void setPlasticity(final float plasticity) {
    if (plasticity < 0 || plasticity > 1) {
      throw new IllegalArgumentException(String.format("Plasticity (%s) must be [0, 1].", plasticity));
    }
    this.plasticity = plasticity;
  }

  @Override
  public void clean() {
    super.clean();
    priorTouches.clean();
  }
}
