package ai.xng;

import java.io.Serializable;
import java.util.Arrays;

import lombok.Getter;
import lombok.val;

public interface Posterior extends Node {
  ThresholdIntegrator getIntegrator();

  @Override
  PosteriorCluster<?> getCluster();

  Connections.Priors getPriors();

  class Trait implements Serializable {
    private final Posterior owner;
    @Getter
    private final Connections.Priors priors;

    public Trait(final Posterior owner) {
      this.owner = owner;
      priors = new Connections.Priors(owner);
    }

    public void activate() {
      final long now = Scheduler.global.now();
      final float plasticity = owner.getCluster()
          .getPlasticity();

      for (val prior : priors) {
        // LTP due to STDP
        prior.edge().distribution.reinforce(prior.node().getTrace().evaluate(now, prior.edge().profile) * plasticity);
      }
    }
  }

  default Posterior conjunction(final Prior... priors) {
    final float coefficient = (ThresholdIntegrator.THRESHOLD + Prior.THRESHOLD_MARGIN) / priors.length;
    if (coefficient * (priors.length - 1) >= ThresholdIntegrator.THRESHOLD) {
      throw new IllegalArgumentException(
          "Too many priors to guarantee reliable conjunction. Recommend staging the evaluation into a tree.");
    }

    for (val prior : priors) {
      prior.getPosteriors().getEdge(this, IntegrationProfile.TRANSIENT).distribution.set(coefficient);
    }

    return this;
  }

  default Posterior disjunction(final Prior... priors) {
    return disjunction(Arrays.asList(priors));
  }

  default Posterior disjunction(final Iterable<? extends Prior> priors) {
    for (val prior : priors) {
      prior.then(this);
    }

    return this;
  }

  default Posterior inhibitor(final Prior prior) {
    return inhibitor(prior, IntegrationProfile.TRANSIENT);
  }

  default Posterior inhibitor(final Prior prior, final IntegrationProfile profile) {
    prior.inhibit(this, profile);
    return this;
  }
}
