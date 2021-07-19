package ai.xng;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Stack;

import org.junit.jupiter.api.Test;

import ai.xng.Cluster.PriorClusterProfile;
import lombok.val;

public class CaptureTest {
  private static final int MAX_PRIORS = 10;

  private final Scheduler scheduler = new TestScheduler();
  {
    Scheduler.global = scheduler;
  }

  private final ActionCluster actions = new ActionCluster();

  private Cluster.CaptureBuilder capture() {
    return new Cluster.CaptureBuilder() {
      @Override
      protected ai.xng.ActionCluster.Node capture(final Iterable<PriorClusterProfile> priors,
          final PosteriorCluster<?> posteriorCluster) {
        return new Cluster.Capture(actions, priors, posteriorCluster).node;
      }
    };
  }

  private ActionCluster.Node capture(final Cluster<? extends Prior> priorCluster,
      final PosteriorCluster<?> posteriorCluster) {
    return capture().priors(priorCluster).posteriors(posteriorCluster);
  }

  @Test
  public void testNoPrior() {
    val monitor = new EmissionMonitor<Long>();
    val input = new InputCluster(), output = new ActionCluster();
    val a = input.new Node(), out = TestUtil.testNode(output, monitor);

    TestUtil.triggerPosterior(out);
    TestUtil.triggerPosterior(capture(input, output));
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());

    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    monitor.reset();

    a.activate();
    scheduler.fastForwardUntilIdle();
    assertFalse(monitor.didEmit());
  }

  @Test
  public void testNoPosterior() {
    val monitor = new EmissionMonitor<Long>();
    val input = new InputCluster(), output = new ActionCluster();
    val a = input.new Node();
    TestUtil.testNode(output, monitor);

    a.activate();
    TestUtil.triggerPosterior(capture(input, output));
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());

    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    monitor.reset();

    a.activate();
    scheduler.fastForwardUntilIdle();
    assertFalse(monitor.didEmit());
  }

  @Test
  public void testAssociate() {
    val input = new InputCluster(),
        output = new ActionCluster();

    val in = input.new Node();
    val monitor = new EmissionMonitor<Long>();
    val out = TestUtil.testNode(output, monitor);

    in.activate();
    TestUtil.triggerPosterior(out);
    TestUtil.triggerPosterior(capture(input, output));
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());

    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    monitor.reset();

    in.activate();
    scheduler.fastForwardUntilIdle();
    assertTrue(monitor.didEmit());
  }

  @Test
  public void testAlreadyAssociated() {
    val input = new InputCluster(),
        output = new ActionCluster();

    val in = input.new Node();
    val monitor = new EmissionMonitor<Long>();
    val out = TestUtil.testNode(output, monitor);

    in.then(out);
    in.activate();
    TestUtil.triggerPosterior(capture(input, output));
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());

    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    monitor.reset();

    in.activate();
    scheduler.fastForwardUntilIdle();
    assertTrue(monitor.didEmit());
  }

  @Test
  public void testDisassociate() {
    val input = new InputCluster(),
        output = new ActionCluster();

    val in = input.new Node();
    val monitor = new EmissionMonitor<Long>();
    val out = TestUtil.testNode(output, monitor);

    in.then(out);
    in.activate();
    TestUtil.inhibitPosterior(out);
    TestUtil.triggerPosterior(capture(input, output));
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());

    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    monitor.reset();

    in.activate();
    scheduler.fastForwardUntilIdle();
    assertFalse(monitor.didEmit());
  }

  @Test
  public void testNoCoincidentAssociate() {
    val input = new InputCluster(),
        output = new ActionCluster();

    val in = input.new Node();
    val monitor = new EmissionMonitor<Long>();
    val out = TestUtil.testNode(output, monitor);

    TestUtil.triggerPosterior(out);
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
    in.activate();
    TestUtil.triggerPosterior(capture(input, output));
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());

    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    monitor.reset();

    in.activate();
    scheduler.fastForwardUntilIdle();
    assertFalse(monitor.didEmit());
  }

  @Test
  public void testCoincidentDisassociate() {
    val input = new InputCluster(),
        output = new ActionCluster();

    val in = input.new Node();
    val monitor = new EmissionMonitor<Long>();
    val out = TestUtil.testNode(output, monitor);

    in.then(out);
    TestUtil.triggerPosterior(out);
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
    in.activate();
    TestUtil.inhibitPosterior(out);
    TestUtil.triggerPosterior(capture(input, output));
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());

    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    monitor.reset();

    in.activate();
    scheduler.fastForwardUntilIdle();
    assertFalse(monitor.didEmit());
  }

  @Test
  public void testConjunction() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    for (int n = 2; n <= MAX_PRIORS; ++n) {
      val input = new InputCluster(),
          output = new ActionCluster();

      val in = new InputNode[n];
      for (int i = 0; i < in.length; ++i) {
        in[i] = input.new Node();
        in[i].activate();
      }

      val monitor = new EmissionMonitor<Long>();
      val out = TestUtil.testNode(output, monitor);
      TestUtil.triggerPosterior(out);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.capture(input, output);

      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
      monitor.reset();

      for (val i : in) {
        i.activate();
      }
      scheduler.fastForwardUntilIdle();
      assertTrue(monitor.didEmit());
    }
  }

  @Test
  public void testAllButOne() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    for (int n = 2; n <= MAX_PRIORS; ++n) {
      val input = new InputCluster(), output = new ActionCluster();

      val in = new InputNode[n];
      for (int i = 0; i < in.length; ++i) {
        in[i] = input.new Node();
        in[i].activate();
      }

      val monitor = new EmissionMonitor<Long>();
      val out = TestUtil.testNode(output, monitor);
      TestUtil.triggerPosterior(out);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.capture(input, output);

      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
      monitor.reset();

      for (int i = 0; i < in.length - 1; ++i) {
        in[i].activate();
      }
      scheduler.fastForwardUntilIdle();
      assertFalse(monitor.didEmit());
    }
  }

  @Test
  public void testTestPriorJitter() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    for (int n = 2; n <= MAX_PRIORS; ++n) {
      val input = new InputCluster(), output = new ActionCluster();

      val in = new InputNode[n];
      for (int i = 0; i < in.length; ++i) {
        in[i] = input.new Node();
        in[i].activate();
        scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.peak());
      }

      val monitor = new EmissionMonitor<Long>();
      val out = TestUtil.testNode(output, monitor);
      TestUtil.triggerPosterior(out);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.capture(input, output);

      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
      monitor.reset();

      for (val i : in) {
        i.activate();
        scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.peak());
      }
      scheduler.fastForwardUntilIdle();
      assertTrue(monitor.didEmit(), String.format("Failed with %s priors.", n));
    }
  }

  @Test
  public void testAllButOneJitter() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    for (int n = 2; n <= MAX_PRIORS; ++n) {
      val input = new InputCluster(), output = new ActionCluster();

      val in = new InputNode[n];
      for (int i = 0; i < in.length; ++i) {
        in[i] = input.new Node();
        in[i].activate();
        scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.peak());
      }

      val monitor = new EmissionMonitor<Long>();
      val out = TestUtil.testNode(output, monitor);
      TestUtil.triggerPosterior(out);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.capture(input, output);

      scheduler.fastForwardFor(IntegrationProfile.PERSISTENT.period());
      monitor.reset();

      for (int i = 0; i < in.length - 1; ++i) {
        in[i].activate();
        scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.peak());
      }
      scheduler.fastForwardUntilIdle();
      assertFalse(monitor.didEmit(), String.format("Failed with %s priors.", n));
    }
  }

  @Test
  public void testLeastSignificantOmitted() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    // This test fails at 4 priors, which is okay as by then the least significant
    // prior will have decayed during training. A previous version with less leeway
    // would not fail until 7 priors.
    for (int n = 2; n <= 3; ++n) {
      val input = new InputCluster(), output = new ActionCluster();

      val in = new InputNode[n];
      for (int i = 0; i < in.length; ++i) {
        in[i] = input.new Node();
        in[i].activate();
        scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.peak());
      }

      val monitor = new EmissionMonitor<Long>();
      val out = TestUtil.testNode(output, monitor);
      TestUtil.triggerPosterior(out);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.capture(input, output);

      scheduler.fastForwardFor(IntegrationProfile.PERSISTENT.period());
      monitor.reset();

      for (int i = 1; i < in.length; ++i) {
        in[i].activate();
        scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.peak());
      }
      scheduler.fastForwardUntilIdle();
      assertFalse(monitor.didEmit(), String.format("Failed with %s priors.", n));
    }
  }

  /**
   * This test should be roughly equivalent to the prior jitter test, but is
   * structured as a causal chain.
   */
  @Test
  public void testStickSequence() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    val input = new InputCluster(), recog = new BiCluster(), output = new ActionCluster();

    val in = input.new Node();
    Prior tail = in;
    for (int i = 0; i < MAX_PRIORS; ++i) {
      tail = tail.then(recog.new Node());
    }

    in.activate();
    scheduler.fastForwardUntilIdle();

    val monitor = new EmissionMonitor<Long>();
    val out = TestUtil.testNode(output, monitor);
    TestUtil.triggerPosterior(out);
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
    Cluster.capture(recog, output);

    scheduler.fastForwardFor(IntegrationProfile.PERSISTENT.period());
    monitor.reset();

    in.activate();
    scheduler.fastForwardUntilIdle();
    assertTrue(monitor.didEmit());
  }

  @Test
  public void testFullyDelayedTraining() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    val input = new InputCluster(), output = new ActionCluster();

    final InputNode in = input.new Node();
    in.activate();

    val monitor = new EmissionMonitor<Long>();
    val out = TestUtil.testNode(output, monitor);
    TestUtil.triggerPosterior(out);
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    Cluster.capture(input, output);

    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    monitor.reset();

    in.activate();
    scheduler.fastForwardUntilIdle();
    assertFalse(monitor.didEmit());
  }

  @Test
  public void testMostlyDelayedTraining() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    val input = new InputCluster(), output = new ActionCluster();

    final InputNode in = input.new Node();
    in.activate();

    val monitor = new EmissionMonitor<Long>();
    val out = TestUtil.testNode(output, monitor);
    TestUtil.triggerPosterior(out);
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period() - 1);
    Cluster.capture(input, output);

    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    monitor.reset();

    in.activate();
    scheduler.fastForwardUntilIdle();
    assertFalse(monitor.didEmit());
  }

  @Test
  public void testDelayedAllButOne() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    for (int n = 1; n <= MAX_PRIORS; ++n) {
      val input = new InputCluster(), output = new ActionCluster();

      val in = new InputNode[n];
      for (int i = 0; i < in.length; ++i) {
        in[i] = input.new Node();
        in[i].activate();
      }

      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period() / 3);

      val monitor = new EmissionMonitor<Long>();
      val out = TestUtil.testNode(output, monitor);
      TestUtil.triggerPosterior(out);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.capture(input, output);

      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
      monitor.reset();

      for (int i = 0; i < in.length - 1; ++i) {
        in[i].activate();
      }
      scheduler.fastForwardUntilIdle();
      assertFalse(monitor.didEmit());
    }
  }

  @Test
  public void testAssociateDisassociateSymmetry() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    val monitor = new EmissionMonitor<Long>();

    val priorCluster = new InputCluster();
    val posteriorCluster = new ActionCluster();
    val prior = priorCluster.new Node();
    val posterior = TestUtil.testNode(posteriorCluster, monitor);

    for (int i = 0; i < 100; ++i) {
      prior.activate();
      TestUtil.triggerPosterior(posterior);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.capture(priorCluster, posteriorCluster);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());

      monitor.reset();
      prior.activate();
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
      assertTrue(monitor.didEmit(), String.format("Failed on iteration %s. Posteriors: %s", i, prior.getPosteriors()));

      prior.activate();
      TestUtil.inhibitPosterior(posterior);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.capture(priorCluster, posteriorCluster);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());

      monitor.reset();
      prior.activate();
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
      assertFalse(monitor.didEmit(), String.format("Failed on iteration %s. Posteriors: %s", i, prior.getPosteriors()));
    }
  }

  // TODO: Use linked structure instead of salience stack. A linked structure
  // seems more correct and may eliminate the need to keep the disassociate
  // operation.
  @Test
  public void testStack() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    val priorCluster = new InputCluster();
    val posteriorCluster = new SignalCluster();
    val testStack = priorCluster.new Node();
    val monitor = EmissionMonitor.fromObservable(posteriorCluster.rxActivations());
    val refStack = new Stack<SignalCluster.Node>();

    for (int i = 0; i < 32; ++i) {
      val item = posteriorCluster.new Node();
      refStack.push(item);

      testStack.activate();
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.scalePosteriors(priorCluster, KnowledgeBase.STACK_FACTOR);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());

      testStack.activate();
      TestUtil.triggerPosterior(item);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.capture(priorCluster, posteriorCluster);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    }

    monitor.reset();

    while (!refStack.isEmpty()) {
      val item = refStack.pop();
      testStack.activate();
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.scalePosteriors(priorCluster, 1 / KnowledgeBase.STACK_FACTOR);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
      Cluster.disassociate(priorCluster, posteriorCluster);
      scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
      assertThat(monitor.emissions()).containsExactly(item);
    }
  }

  /**
   * Ensures that any residual connection after a symmetric associate/disassociate
   * pair will not be scaled back up to potency during stack pops.
   */
  @Test
  public void testStackEviction() {
    val scheduler = new TestScheduler();
    Scheduler.global = scheduler;

    val monitor = new EmissionMonitor<Long>();

    val priorCluster = new InputCluster();
    val posteriorCluster = new ActionCluster();
    val prior = priorCluster.new Node();
    val posterior = TestUtil.testNode(posteriorCluster, monitor);

    // An associate/disassociate pair from testAssociateDisassociateSymmetry before
    // scaling.

    prior.activate();
    TestUtil.triggerPosterior(posterior);
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.defaultInterval());
    Cluster.capture(priorCluster, posteriorCluster);
    scheduler.fastForwardFor(IntegrationProfile.PERSISTENT.period());

    prior.activate();
    scheduler.fastForwardFor(2 * IntegrationProfile.TRANSIENT.defaultInterval());
    Cluster.disassociate(priorCluster, posteriorCluster);
    scheduler.fastForwardFor(IntegrationProfile.PERSISTENT.period());

    prior.activate();
    scheduler.fastForwardFor(IntegrationProfile.TRANSIENT.period());
    Cluster.scalePosteriors(priorCluster, (float) Math.pow(1 / KnowledgeBase.STACK_FACTOR, 32));
    scheduler.fastForwardFor(IntegrationProfile.PERSISTENT.period());

    monitor.reset();
    prior.activate();
    scheduler.fastForwardUntilIdle();
    assertFalse(monitor.didEmit());
  }
}
