package ai.xng;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiConsumer;

import com.google.common.collect.ImmutableList;

import ai.xng.constructs.CoincidentEffect;
import io.reactivex.Observable;
import io.reactivex.subjects.PublishSubject;
import io.reactivex.subjects.Subject;
import lombok.val;

public abstract class Cluster<T extends Node> implements Serializable {
  private final WeakSerializableRecencyQueue<T> activations = new WeakSerializableRecencyQueue<>();
  private transient Subject<T> rxActivations;

  public Observable<T> rxActivations() {
    return rxActivations;
  }

  public Cluster() {
    init();
  }

  // TODO: register cleanup task

  private void readObject(final ObjectInputStream o) throws ClassNotFoundException, IOException {
    o.defaultReadObject();
    init();
  }

  private void init() {
    rxActivations = PublishSubject.create();
  }

  protected class ClusterNodeTrait implements Serializable {
    private final WeakSerializableRecencyQueue<T>.Link link;

    public ClusterNodeTrait(final T owner) {
      link = activations.new Link(owner);
    }

    /**
     * Promotes the node within the recency queue and publishes it to the
     * Observable. This should be called at the beginning of the node activation to
     * ensure that any side effects that query the recency queue receive a sequence
     * consistent with the updated activation timestamps.
     */
    public void activate() {
      link.promote();
      rxActivations.onNext(link.get());
    }
  }

  public void clean() {
    activations.clean();
  }

  public Iterable<T> activations() {
    return activations::iterator;
  }

  public static <T extends Node> void forEachByTrace(final Cluster<? extends T> cluster,
      final IntegrationProfile profile,
      final long t, final BiConsumer<T, Float> action) {
    final long horizon = t - profile.period();

    for (final T node : cluster.activations) {
      if (node.getLastActivation().get() <= horizon) {
        break;
      }

      final float trace = node.getTrace().evaluate(t, profile);
      if (trace > 0) {
        action.accept(node, trace);
      }
    }
  }

  public static record PriorClusterProfile(Cluster<? extends Prior> cluster,
      ImmutableList<IntegrationProfile> profiles) {

    public PriorClusterProfile(final Cluster<? extends Prior> cluster, final IntegrationProfile... profiles) {
      this(cluster, ImmutableList.copyOf(profiles));
    }

    public static class ListBuilder {
      private final ImmutableList.Builder<PriorClusterProfile> backing = ImmutableList.builder();
      private ImmutableList<IntegrationProfile> baseProfiles = ImmutableList.of(IntegrationProfile.TRANSIENT);

      public ListBuilder baseProfiles(final IntegrationProfile... profiles) {
        baseProfiles = ImmutableList.copyOf(profiles);
        return this;
      }

      public ListBuilder add(final Cluster<? extends Prior> cluster, final IntegrationProfile... additionalProfiles) {
        backing.add(
            new PriorClusterProfile(cluster,
                ImmutableList.<IntegrationProfile>builder()
                    .addAll(baseProfiles)
                    .addAll(Arrays.asList(additionalProfiles))
                    .build()));
        return this;
      }

      public ImmutableList<PriorClusterProfile> build() {
        return backing.build();
      }
    }
  }

  public static abstract class CaptureBuilder {
    private final PriorClusterProfile.ListBuilder priors = new PriorClusterProfile.ListBuilder();

    public CaptureBuilder baseProfiles(final IntegrationProfile... profiles) {
      priors.baseProfiles(profiles);
      return this;
    }

    public CaptureBuilder priors(final Cluster<? extends Prior> cluster,
        final IntegrationProfile... additionalProfiles) {
      priors.add(cluster, additionalProfiles);
      return this;
    }

    public ActionCluster.Node posteriors(final PosteriorCluster<?> posteriorCluster) {
      return capture(priors.build(), posteriorCluster);
    }

    protected abstract ActionCluster.Node capture(Iterable<PriorClusterProfile> priors,
        PosteriorCluster<?> posteriorCluster);
  }

  /**
   * Captures a posterior activation state to be reproduced by activations in the
   * given prior clusters, using the given integration profiles. Posteriors are
   * captured by coincidence and priors are captured by trace.
   * <p>
   * To respect firing order, the traces for priors are evaluated against the
   * firing time of each posterior. However, to ensure that the expected
   * disassociation happens for any edge not exercised during the capture, this
   * occurs over the set of priors captured during this operation rather than the
   * set of priors with nonzero traces at the time of posterior activation.
   * Notably this means that priors whose traces have decayed by the time the
   * capture takes place are ignored even if they did have nonzero traces at the
   * time of posterior activation. This should not affect common usage.
   * <p>
   * More sophisticated capture behaviors are conceivable but requirements easily
   * become self inconsistent.
   */
  public static class Capture extends CoincidentEffect<Posterior> {
    private final Iterable<PriorClusterProfile> priors;

    private transient ConjunctionJunction capturedPriors;
    private transient Set<Posterior> capturedPosteriors;

    public Capture(final ActionCluster actionCluster, final Iterable<PriorClusterProfile> priors,
        final PosteriorCluster<?> posteriorCluster) {
      super(actionCluster, posteriorCluster);
      this.priors = priors;
    }

    @Override
    protected void onActivate() {
      capturePriors();
      scheduleDeactivationCheck();
      capturedPosteriors = new HashSet<>();
      super.onActivate();
    }

    private void capturePriors() {
      final long t = Scheduler.global.now();
      capturedPriors = new ConjunctionJunction();

      for (val prior : priors) {
        for (val profile : prior.profiles) {
          forEachByTrace(prior.cluster, profile, t,
              (node, trace) -> capturedPriors.add(node, profile, trace));
        }
      }
    }

    private void scheduleDeactivationCheck() {
      final Optional<Long> deactivation = node.getIntegrator().nextThreshold(Scheduler.global.now(), -1);
      if (deactivation.isPresent()) {
        Scheduler.global.postTask(() -> {
          if (node.getIntegrator().isActive()) {
            scheduleDeactivationCheck();
          } else {
            endCapture();
          }
        }, deactivation.get());
      } else {
        endCapture();
      }
    }

    /**
     * Disassociates any posteriors that were not active during the capture.
     */
    private void endCapture() {
      for (val prior : capturedPriors) {
        for (val posterior : prior.node().getPosteriors()) {
          if (!capturedPosteriors.contains(posterior.node())) {
            val distribution = posterior.edge().distribution;
            distribution.reinforce(weightByTrace(-distribution.getWeight(), 0, prior.weight()));
          }
        }
      }

      capturedPriors = null;
      capturedPosteriors = null;
    }

    @Override
    protected void apply(final Posterior posterior) {
      // Re-evaluate prior traces to respect firing order.
      final long t = posterior.getLastActivation().get();
      val timeSensitiveConjunction = new ConjunctionJunction();
      for (val component : capturedPriors) {
        timeSensitiveConjunction.add(component.node(), component.profile(),
            component.node().getTrace().evaluate(t, component.profile()));
      }
      timeSensitiveConjunction.build(posterior);
      capturedPosteriors.add(posterior);
    }
  }

  private static float weightByTrace(final float value, final float identity, final float trace) {
    final float tolerantTrace = Math.min(1, trace * (1 + Prior.THRESHOLD_MARGIN));
    return identity + (value - identity) * tolerantTrace;
  }

  public static void scalePosteriors(final Cluster<? extends Prior> priorCluster, final float factor) {
    forEachByTrace(priorCluster, IntegrationProfile.TRANSIENT, Scheduler.global.now(),
        (prior, trace) -> {
          for (val entry : prior.getPosteriors()) {
            entry.edge().distribution.scale(weightByTrace(factor, 1, trace));
          }
        });
  }
}
