package ai.xng;

public class SignalCluster extends PosteriorCluster<SignalCluster.Node> {
  public class Node extends OutputNode {
    private final ClusterNodeTrait clusterNode;

    public Node() {
      clusterNode = new ClusterNodeTrait(this);
    }

    @Override
    public SignalCluster getCluster() {
      return SignalCluster.this;
    }

    @Override
    public ThresholdIntegrator getIntegrator() {
      return clusterNode.getIntegrator();
    }

    @Override
    public final void activate() {
      clusterNode.activate();
      super.activate();
    }
  }
}
