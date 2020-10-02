package ai.xng;

import java.io.Serializable;
import java.util.Map;

import lombok.RequiredArgsConstructor;

public abstract class ActionNode implements Posterior {
  private static final long serialVersionUID = 1L;

  private final Posterior.Trait input = new Posterior.Trait(this);

  @Override
  public ThresholdIntegrator getIntegrator() {
    return input.getIntegrator();
  }

  @Override
  public Map<Prior, Distribution> getPriors() {
    return input.getPriors();
  }

  @Override
  public void activate() {
    input.activate();
  }

  @RequiredArgsConstructor
  public static abstract class Lambda extends ActionNode {
    private static final long serialVersionUID = 1L;

    /**
     * Convenience functional interface. Since each {@code ActionNode} is likely to
     * have a different {@code activate} implementation, this allows us to avoid
     * needing to declare serial version UIDs for all of them.
     */
    @FunctionalInterface
    public static interface OnActivate extends Serializable {
      void run();
    }

    private final OnActivate onActivate;

    @Override
    public void activate() {
      onActivate.run();
      super.activate();
    }
  }
}