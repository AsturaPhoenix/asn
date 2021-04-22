package ai.xng;

public class DataCluster extends PosteriorCluster<DataCluster.Node> {
  private final InputCluster updateCluster;

  public abstract class Node extends OutputNode implements DataNode {
    private final ClusterNodeTrait clusterNode = new ClusterNodeTrait(this);

    private Node() {
    }

    @Override
    public DataCluster getCluster() {
      return DataCluster.this;
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

  public class FinalNode<T> extends Node {
    private final SerializableOrProxy<T> container;

    @Override
    public T getData() {
      return container.getData();
    }

    public FinalNode(final T value) {
      container = new SerializableOrProxy<>(value);
    }

    @Override
    public String toString() {
      return "final " + getData();
    }
  }

  public class MutableNode<T> extends Node {
    private final DataNode.MaybeTransient<T> container = new DataNode.MaybeTransient<>();

    /**
     * Node activated when {@link #setData(Object)} is called. This node is
     * activated whether or not the data was actually changed.
     * <p>
     * This node is activated directly, so rapid updates are subject to temporal
     * summation. Conjunctions and other posteriors sensitive to temporal summation
     * should buffer with an intermediary node. .
     */
    public final InputCluster.Node onUpdate = updateCluster.new Node();

    @Override
    public T getData() {
      return container.data;
    }

    public void setData(final T value) {
      container.data = value;
      onUpdate.activate();
    }

    @Override
    public String toString() {
      return "mutable " + getData();
    }
  }

  public DataCluster(final InputCluster updateCluster) {
    this.updateCluster = updateCluster;
  }
}
