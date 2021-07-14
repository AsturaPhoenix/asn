package ai.xng;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import com.google.common.collect.ImmutableSet;

import ai.xng.Cluster.PriorClusterProfile;
import ai.xng.constructs.CoincidentEffect;
import io.reactivex.Observable;
import io.reactivex.subjects.PublishSubject;
import io.reactivex.subjects.Subject;
import lombok.val;

public class KnowledgeBase implements Serializable, AutoCloseable {
  public static final float STACK_FACTOR = .5f, PUSH_FACTOR = STACK_FACTOR, POP_FACTOR = 1 / STACK_FACTOR;

  private transient Subject<String> rxOutput;

  public final InputCluster input = new InputCluster();
  public final DataCluster data = new DataCluster(input);
  public final DataCluster.MutableNode<String> inputValue = data.new MutableNode<>();
  public final DataCluster.MutableNode<Throwable> lastException = data.new MutableNode<>();
  public final DataCluster.MutableNode<Object> returnValue = data.new MutableNode<>();

  public final BiCluster stateRecognition = new BiCluster("stateRecognition"),
      sequenceRecognition = new BiCluster("sequenceRecognition"),
      context = new BiCluster("context"),
      naming = new BiCluster("naming"),
      entrypoint = new BiCluster("entrypoint"),
      execution = new BiCluster("execution");
  public final ActionCluster actions = new ActionCluster(lastException);
  public final SignalCluster signals = new SignalCluster();

  public final SignalCluster.Node variadicEnd = signals.new Node();

  public final ActionCluster.Node print = new CoincidentEffect.Lambda<>(actions, data,
      node -> rxOutput.onNext(Objects.toString(node.getData()))).node;

  public Observable<String> rxOutput() {
    return rxOutput;
  }

  private final Map<BiCluster, ActionCluster.Node> suppressPosteriors = new HashMap<>();

  public ActionCluster.Node suppressPosteriors(final BiCluster cluster) {
    return suppressPosteriors.computeIfAbsent(cluster, key -> new CoincidentEffect.Lambda<>(actions, key, node -> {
      // TODO: Isn't this always 1?
      final float coincidence = node.getIntegrator().getNormalizedCappedValue();
      assert coincidence == 1;
      for (final Connections.Entry<Posterior> entry : node.getPosteriors()) {
        entry.edge().suppress(coincidence);
      }
    }).node);
  }

  private static record ScalePosteriorsKey(Cluster<? extends Prior> cluster, float factor) implements Serializable {
  }

  private final Map<ScalePosteriorsKey, ActionCluster.Node> scalePosteriors = new HashMap<>();

  public ActionCluster.Node scalePosteriors(final Cluster<? extends Prior> cluster, final float factor) {
    return scalePosteriors.computeIfAbsent(new ScalePosteriorsKey(cluster, factor),
        key -> actions.new Node(() -> Cluster.scalePosteriors(key.cluster, key.factor)));
  }

  private final Map<Cluster<? extends Prior>, ActionCluster.Node> clearPosteriors = new HashMap<>();

  public ActionCluster.Node clearPosteriors(final Cluster<? extends Prior> cluster) {
    return clearPosteriors.computeIfAbsent(cluster, key -> actions.new Node(() -> Cluster.disassociateAll(key)));
  }

  private static record ResetPosteriorsKey(Cluster<? extends Prior> cluster, boolean outboundOnly) {
  }

  private final Map<ResetPosteriorsKey, ActionCluster.Node> resetPosteriors = new HashMap<>();

  /**
   * Resets the posteriors and traces of recently activated nodes in a prior
   * cluster.
   */
  private ActionCluster.Node resetPosteriors(final Cluster<? extends Prior> cluster, final boolean outboundOnly) {
    return resetPosteriors.computeIfAbsent(new ResetPosteriorsKey(cluster, outboundOnly),
        key -> actions.new Node(() -> {
          final long now = Scheduler.global.now();
          final long horizon = now - IntegrationProfile.PERSISTENT.period();

          for (val node : key.cluster.activations()) {
            if (node.getLastActivation().get() <= horizon) {
              break;
            }

            for (val entry : node.getPosteriors()) {
              if (!key.outboundOnly || entry.node().getCluster() != key.cluster) {
                entry.edge().clearSpikes();
              }
            }
            node.getTrace().evict(now);
          }
        }));
  }

  public ActionCluster.Node resetPosteriors(final Cluster<? extends Prior> cluster) {
    return resetPosteriors(cluster, false);
  }

  public ActionCluster.Node resetOutboundPosteriors(final Cluster<? extends Prior> cluster) {
    return resetPosteriors(cluster, true);
  }

  private final Map<Cluster<? extends Posterior>, ActionCluster.Node> resetPriors = new HashMap<>();

  /**
   * Resets the priors of recently activated nodes in a posterior cluster.
   */
  public ActionCluster.Node resetPriors(final Cluster<? extends Posterior> cluster) {
    return resetPriors.computeIfAbsent(cluster,
        key -> actions.new Node(() -> {
          final long now = Scheduler.global.now();
          final long horizon = now - IntegrationProfile.PERSISTENT.period();

          for (val node : key.activations()) {
            if (node.getLastActivation().get() <= horizon) {
              break;
            }

            for (val entry : node.getPriors()) {
              entry.edge().clearSpikes();
            }
          }
        }));
  }

  private static record AssociateKey(Set<PriorClusterProfile> priors, PosteriorCluster<?> posteriorCluster)
      implements Serializable {
  }

  private final Map<AssociateKey, ActionCluster.Node> capture = new HashMap<>();

  public ActionCluster.Node capture(final Iterable<PriorClusterProfile> priors,
      final PosteriorCluster<?> posteriorCluster) {
    return capture.computeIfAbsent(new AssociateKey(ImmutableSet.copyOf(priors), posteriorCluster),
        key -> actions.new Node(() -> Cluster.capture(key.priors, key.posteriorCluster)));
  }

  public Cluster.AssociationBuilder<ActionCluster.Node> capture() {
    return new Cluster.AssociationBuilder<>() {
      @Override
      protected ActionCluster.Node associate(final Iterable<PriorClusterProfile> priors,
          final PosteriorCluster<?> posteriorCluster) {
        return KnowledgeBase.this.capture(priors, posteriorCluster);
      }
    };
  }

  public ActionCluster.Node capture(final Cluster<? extends Prior> priorCluster,
      final PosteriorCluster<?> posteriorCluster) {
    return capture().priors(priorCluster).to(posteriorCluster);
  }

  private final Map<AssociateKey, ActionCluster.Node> associate = new HashMap<>();

  @Deprecated
  public ActionCluster.Node associate(final Iterable<PriorClusterProfile> priors,
      final PosteriorCluster<?> posteriorCluster) {
    return associate.computeIfAbsent(new AssociateKey(ImmutableSet.copyOf(priors), posteriorCluster),
        key -> actions.new Node(() -> Cluster.associate(key.priors, key.posteriorCluster)));
  }

  @Deprecated
  public Cluster.AssociationBuilder<ActionCluster.Node> associate() {
    return new Cluster.AssociationBuilder<>() {
      @Override
      protected ActionCluster.Node associate(final Iterable<PriorClusterProfile> priors,
          final PosteriorCluster<?> posteriorCluster) {
        return KnowledgeBase.this.associate(priors, posteriorCluster);
      }
    };
  }

  @Deprecated
  public ActionCluster.Node associate(final Cluster<? extends Prior> priorCluster,
      final PosteriorCluster<?> posteriorCluster) {
    return associate().priors(priorCluster).to(posteriorCluster);
  }

  private static record DisassociateKey(Cluster<? extends Prior> priorCluster, PosteriorCluster<?> posteriorCluster)
      implements Serializable {
  }

  private final Map<DisassociateKey, ActionCluster.Node> disassociate = new HashMap<>();

  public ActionCluster.Node disassociate(final Cluster<? extends Prior> priorCluster,
      final PosteriorCluster<?> posteriorCluster) {
    return disassociate.computeIfAbsent(new DisassociateKey(priorCluster, posteriorCluster),
        key -> actions.new Node(() -> Cluster.disassociate(key.priorCluster, key.posteriorCluster)));
  }

  public KnowledgeBase() {
    init();
  }

  private void init() {
    rxOutput = PublishSubject.create();
  }

  private void readObject(final ObjectInputStream stream) throws ClassNotFoundException, IOException {
    stream.defaultReadObject();
    init();
  }

  @Override
  public void close() {
    rxOutput.onComplete();
  }
}
