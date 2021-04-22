package ai.xng;

import java.io.IOException;
import java.io.ObjectInputStream;

import lombok.Getter;
import lombok.val;

public abstract class PosteriorCluster<T extends Posterior> extends Cluster<T> {
  public static final float DEFAULT_PLASTICITY = .1f;

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

  public Iterable<T> priorTouches() {
    return priorTouches::iterator;
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
