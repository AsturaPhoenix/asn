package ai.xng;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.function.BiConsumer;

import com.google.common.collect.ImmutableList;

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

  public static <T extends Node> void forEachByTrace(final Cluster<T> cluster, final IntegrationProfile profile,
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

  private static ConjunctionJunction priorConjunction(final Iterable<PriorClusterProfile> priors, final long t) {
    val conjunction = new ConjunctionJunction();
    for (val prior : priors) {
      for (val profile : prior.profiles) {
        forEachByTrace(prior.cluster, profile, t, (node, trace) -> conjunction.add(node, profile, trace));
      }
    }
    return conjunction;
  }

  private static void associate(final Iterable<PriorClusterProfile> priors, final Posterior posterior,
      final long t, final float weight) {
    priorConjunction(priors, t).build(posterior, (distribution, coefficient) -> {
      // This condition prevents association from ever making pre-existing connections
      // more restrictive.
      if (coefficient >= distribution.getMode()) {
        distribution.add(coefficient, weight);
      }
    });
  }

  /**
   * Forms explicit conjunctive associations between the given prior cluster and
   * posterior clusters, using the given integration profiles. The active
   * posteriors considered at the time this function executes are always based on
   * a {@link IntegrationProfile#TRANSIENT} window.
   */
  @Deprecated
  public static void associate(final Iterable<PriorClusterProfile> priors,
      final PosteriorCluster<?> posteriorCluster) {
    forEachByTrace(posteriorCluster, IntegrationProfile.TRANSIENT, Scheduler.global.now(),
        (posterior, posteriorTrace) -> {
          associate(priors, posterior, posterior.getLastActivation().get(), posteriorTrace);
        });
  }

  /**
   * Captures posterior activation states to be reproduced by activations in the
   * given prior clusters, using the given integration profiles. Only active
   * posteriors are allowed to pull up coefficients, and only inactive posteriors
   * are allowed to pull them down. (This includes inactive posteriors that are
   * not yet connected, to capture inhibition.) Pull-up or pull-down is done by
   * distributing the integration difference across the conjuncted priors.
   * Integration difference is the difference between the actual observed
   * posterior integration and the computed contribution of the priors from their
   * integration profiles and activation traces. Individual adjustments are made
   * by taking weight from the mode and moving it to the new setpoint.
   * <p>
   * Furthermore, to incorporate STDP, the integration state at the time of prior
   * activation is subtracted from the integration state at the time of capture.
   * If the difference changes sign once adjusted this way, it is discarded.
   */
  public static void capture(final Iterable<PriorClusterProfile> priors,
      final PosteriorCluster<?> posteriorCluster) {
    final long t = Scheduler.global.now();

    // First, pre-build a conjunction junction to capture the relative contributions
    // of the selected priors, before we start looping over posteriors. This will be
    // used to distribute any weight changes.
    val conjunction = priorConjunction(priors, t);

    for (final Posterior posterior : posteriorCluster.priorTouches()) {
      val priorContribution = new float[1];
      for (val prior : priors) {
        for (val profile : prior.profiles) {
          forEachByTrace(prior.cluster, profile, t,
              (node, trace) -> priorContribution[0] += node.getPosteriors().getEdge(posterior, profile).distribution
                  .getMode() * trace);
        }
      }

      val integrator = posterior.getIntegrator();
      final float rawResidual = integrator.getValue() - priorContribution[0];
      final float adjustedResidual = rawResidual - integrator.getValue();

      // Active posteriors are allowed to pull up priors while inactive posteriors are
      // allowed to pull them down.
      if (integrator.isActive() && residual > 0 || !integrator.isActive() && residual < 0) {
        // For simplificity, we don't make any effort to establish an intelligent
        // conjunctive margin and instead capture the posterior as-is. Additionally, due
        // to time quantization, note that the posterior may be superthreshold even if
        // it should naively be merely at threshold.
        conjunction.build(posterior, 0,
            (distribution, coefficient) -> distribution.move(distribution.getMode() + residual * coefficient, 1));
      }
    }
  }

  public static abstract class AssociationBuilder<T> {
    private final PriorClusterProfile.ListBuilder priors = new PriorClusterProfile.ListBuilder();

    public AssociationBuilder<T> baseProfiles(final IntegrationProfile... profiles) {
      priors.baseProfiles(profiles);
      return this;
    }

    public AssociationBuilder<T> priors(final Cluster<? extends Prior> cluster,
        final IntegrationProfile... additionalProfiles) {
      priors.add(cluster, additionalProfiles);
      return this;
    }

    public T to(final PosteriorCluster<?> posteriorCluster) {
      return associate(priors.build(), posteriorCluster);
    }

    protected abstract T associate(Iterable<PriorClusterProfile> priors, PosteriorCluster<?> posteriorCluster);
  }

  public static abstract class ChainableAssociationBuilder extends AssociationBuilder<ChainableAssociationBuilder> {
  }

  @Deprecated
  public static ChainableAssociationBuilder associate() {
    return new ChainableAssociationBuilder() {
      @Override
      protected ChainableAssociationBuilder associate(final Iterable<PriorClusterProfile> priors,
          final PosteriorCluster<?> posteriorCluster) {
        Cluster.associate(priors, posteriorCluster);
        return this;
      }
    };
  }

  @Deprecated
  public static void associate(final Cluster<? extends Prior> priorCluster,
      final PosteriorCluster<?> posteriorCluster) {
    associate().priors(priorCluster).to(posteriorCluster);
  }

  public static ChainableAssociationBuilder capture() {
    return new ChainableAssociationBuilder() {
      @Override
      protected ChainableAssociationBuilder associate(final Iterable<PriorClusterProfile> priors,
          final PosteriorCluster<?> posteriorCluster) {
        capture(priors, posteriorCluster);
        return this;
      }
    };
  }

  public static void capture(final Cluster<? extends Prior> priorCluster, final PosteriorCluster<?> posteriorCluster) {
    capture().priors(priorCluster).to(posteriorCluster);
  }

  private static float weightByTrace(final float value, final float identity, final float trace) {
    final float tolerantTrace = Math.min(1, trace * (1 + Prior.THRESHOLD_MARGIN));
    return identity + (value - identity) * tolerantTrace;
  }

  /**
   * Breaks associations between the given prior cluster and posterior cluster.
   * The active posteriors considered at the time this function executes are
   * always based on a {@link IntegrationProfile#TRANSIENT} window. The degree to
   * which associations are broken are determined by this window and the prior
   * trace.
   */
  @Deprecated
  public static void disassociate(final Cluster<? extends Prior> priorCluster,
      final Cluster<? extends Posterior> posteriorCluster) {
    forEachByTrace(posteriorCluster, IntegrationProfile.TRANSIENT, Scheduler.global.now(),
        (posterior, posteriorTrace) -> {
          // For each posterior, find all priors in the designated cluster and reduce
          // their weight by the product of the pertinent traces.

          for (val prior : posterior.getPriors()) {
            if (prior.node().getCluster() == priorCluster) {
              final float priorTrace = prior.node().getTrace()
                  .evaluate(posterior.getLastActivation().get(), prior.edge().profile);
              if (priorTrace > 0) {
                prior.edge().distribution.reinforce(weightByTrace(
                    -prior.edge().distribution.getWeight(), 0, priorTrace * posteriorTrace));
              }
            }
          }
        });
  }

  public static void disassociateAll(final Cluster<? extends Prior> priorCluster) {
    // This is a reinforcement rather than a simple clear to smooth by trace.
    forEachByTrace(priorCluster, IntegrationProfile.TRANSIENT, Scheduler.global.now(),
        (prior, trace) -> {
          for (val entry : prior.getPosteriors()) {
            entry.edge().distribution
                .reinforce(weightByTrace(-entry.edge().distribution.getWeight(), 0, trace));
          }
        });
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
