package ai.xng;

import java.util.ArrayList;
import java.util.List;

import lombok.RequiredArgsConstructor;
import lombok.val;

@RequiredArgsConstructor
public class ConjunctionJunction {
  private static record WeightedPrior(Prior prior, float weight) {
  }

  private final List<WeightedPrior> priors = new ArrayList<>();
  private final long t;

  private float norm;

  public void add(final Prior prior) {
    final float w = prior.getTrace()
        .evaluate(t)
        .value();
    if (w > 0) {
      priors.add(new WeightedPrior(prior, w));
      norm += w;
    }
  }

  public void build(final Posterior posterior, final float weight) {
    // Scale such that activation of the last principal component (may be
    // hypothetical, with relative weight 1) has margins on either side of the
    // activation threshold.
    final float normAdj = norm - .5f / norm;

    for (val weightedPrior : priors) {
      final float coefficient = Math.min(1, weightedPrior.weight()) / normAdj;
      posterior.getPriors()
          .put(weightedPrior.prior(), new UnimodalHypothesis(coefficient, weight));
    }
  }
}