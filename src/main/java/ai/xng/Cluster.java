package ai.xng;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiConsumer;

import com.google.common.collect.ImmutableList;

import ai.xng.constructs.CoincidentEffect;
import io.reactivex.Observable;
import io.reactivex.subjects.PublishSubject;
import io.reactivex.subjects.Subject;
import lombok.RequiredArgsConstructor;
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

  private static void associate(final Iterable<PriorClusterProfile> priors, final Posterior posterior,
      final long t, final float weight) {
    val conjunction = new ConjunctionJunction();

    for (val prior : priors) {
      for (val profile : prior.profiles) {
        forEachByTrace(prior.cluster, profile, t, (node, trace) -> conjunction.add(node, profile, trace));
      }
    }

    conjunction.build(posterior, (distribution, coefficient) -> {
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
   * Captures a posterior activation state to be reproduced by activations in the
   * given prior clusters, using the given integration profiles. Posteriors are
   * captured by coincidence and priors are captured by trace.
   * <p>
   * The target activation level for a posterior is the default margin on either
   * side of the threshold depending on its state during the capture. The
   * difference between the current activation level caused by the selected priors
   * and the target activation level is distributed according to the inverse of
   * the connection weights. Lower incumbent weights are given the greatest
   * difference so as to maximize the desired effect. If new priors are added to a
   * well established set of priors for a posterior but render that posterior
   * inactive at the time of capture, the new priors are assigned inhibitory
   * weights.
   * <p>
   * A binary target activation level is used rather than pulling up or down
   * towards the activation level at the time of capture to allow for more margin
   * of error in the timing of the capture with respect to the posterior
   * activation. For example, to capture peak activation, the capture would
   * otherwise have to be done at exactly the peak of the posterior activation,
   * which is fragile. In the inactive case, this could also capture surprise
   * residual activation levels that would not be expressed until later.
   * <p>
   * Furthermore, it is difficult to define a reasonable way to capture activation
   * level that respects spike timing. Simply capturing the activation level is
   * likely to capture a decaying "posterior" that may actually have preceded its
   * "priors".
   * <p>
   * The timing of this operation centers around the posteriors. After all, the
   * priors may have heterogeneous timing profiles, so they may well have been
   * activated at different times. This condsideration is asymmetric in that we
   * cannot simply consider the consequences of all priors activated at any given
   * time since summation occurs on the posterior integrators.
   * <p>
   * However, we must consider not only the posteriors that are active (and which
   * therefore have activation times) but also the posteriors that are not active.
   * We must therefore precisely define what it means for a posterior to have been
   * active or inactive in the scope of this operation, preferably over some time
   * period for robustness, and with a fuzzing envelope that smooths cases where a
   * posterior activation falls near a boundary.
   * <p>
   * While a fuzzing envelope is desirable, it opens up too many complications to
   * be worthwhile. Instead, the set of active posteriors is determined solely by
   * coincidence. This formulation makes it desirable to transition posteriors
   * from an inactive to active state lazily, so that "disassociation" happens
   * eagerly, while subsequent activation restores any weakened connections.
   * Unfortunately, it is not obvious how to make disassociation operations
   * reliably reversible in this way, so instead, disassociation will not happen
   * until the end of the capture window.
   * 
   * <h3>Example: heterogeneous timing profiles.</h3>
   * <p>
   * For a somewhat complex practical example, consider sequence capture, which
   * captures a posterior with both short-decay and long-decay priors (in
   * different prior clusters). When such a capture is performed, even though some
   * nodes in the long-decay prior cluster may have been recently activated (in a
   * way consistent with, say, the short-decay timing profile), those priors would
   * not participate in a signficant way as priors in this capture as their traces
   * in the long-decay profile would not be significant. This means those nodes
   * are neither associated with the new posterior, nor are their existing
   * posteriors disassociated.
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
      capturedPriors = null;
      capturedPosteriors = null;
    }

    @Override
    protected void apply(final Posterior posterior) {
      // We need to determine a suitable margin to use to calculate the delta that we
      // will end up distributing amongst the priors according to inverse weight. The
      // ideal margin is the maximum margin that will result in an activation only if
      // all priors are active with timing similar to that captured.

      final long t = Scheduler.global.now();

      val distributor = new Distributor.Inverse();
      float priorContribution = 0;
      final float[] delta = new float[1];

      capturedPriors.build(posterior, 0, (prior, weight) -> {
        val edge = prior.node().getPosteriors().getEdge(posterior, prior.profile()).distribution;
        priorContribution += edge.getMode() * prior.node().getTrace().evaluate(t, prior.profile());
        // TODO: weight = trace
        distributor.add(edge.getWeight(), proportion -> edge.add(proportion * delta[0], weight));
      });

      delta[0] = Prior.DEFAULT_COEFFICIENT - priorContribution;

      distributor.distribute();

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
