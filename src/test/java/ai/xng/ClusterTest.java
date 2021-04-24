package ai.xng;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.Serializable;

import org.junit.jupiter.api.Test;

import lombok.val;

public class ClusterTest {
  @Test
  public void testActivations0() {
    val cluster = new InputCluster();
    cluster.new Node();
    assertThat(cluster.activations()).isEmpty();
  }

  @Test
  public void testActivations3() {
    val cluster = new InputCluster();
    val a = cluster.new Node(), b = cluster.new Node(), c = cluster.new Node();
    a.activate();
    b.activate();
    c.activate();
    assertThat(cluster.activations()).containsExactly(c, b, a);
  }

  private static class SingleNodeCluster implements Serializable {
    final InputCluster cluster = new InputCluster();
    final InputCluster.Node node = cluster.new Node();
  }

  @Test
  public void testActivationsClearedAfterDeserialization() throws Exception {
    val original = new SingleNodeCluster();
    original.node.activate();

    val deserialized = TestUtil.serialize(original);
    assertThat(original.cluster.activations()).containsExactly(original.node);
    assertThat(deserialized.cluster.activations()).isEmpty();

    deserialized.node.activate();
    assertThat(deserialized.cluster.activations()).containsExactly(deserialized.node);
  }

  @Test
  public void testPriorTouches() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    val inputCluster = new InputCluster();
    val outputCluster = new SignalCluster();
    val input = inputCluster.new Node();
    val output = outputCluster.new Node();
    input.getPosteriors().getEdge(output, IntegrationProfile.TRANSIENT).distribution.set(.7f);
    assertThat(outputCluster.priorTouches()).isEmpty();

    input.activate();
    assertThat(outputCluster.priorTouches()).containsExactly(output);

    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period() - 1);
    assertThat(outputCluster.priorTouches()).containsExactly(output);

    scheduler.fastForwardFor(1);
    assertThat(outputCluster.priorTouches()).isEmpty();
  }

  private static class GcTestClusters implements Serializable {
    final InputCluster input = new InputCluster();
    final BiCluster intermediate = new BiCluster();
    final ActionCluster output = new ActionCluster();
  }

  /**
   * Clusters should not hold strong references to anonymous nodes since if the
   * prior cannot be activated, the chain should not have any effect.
   */
  @Test
  public void testGc() throws Exception {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;
    val clusters = new GcTestClusters();

    val gc = new GcFixture(clusters);

    for (int i = 0; i < 1000; ++i) {
      val input = clusters.input.new Node();
      input.then(clusters.intermediate.new Node())
          .then(clusters.output.new Node(() -> {
          }));
      input.activate();
    }

    scheduler.fastForwardUntilIdle();

    gc.assertNoGrowth(() -> {
      System.gc();
      clusters.input.clean();
      clusters.intermediate.clean();
      clusters.output.clean();
    });
  }
}
