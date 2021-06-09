package ai.xng;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import lombok.SneakyThrows;
import lombok.experimental.UtilityClass;

@UtilityClass
public class TestUtil {
  /**
   * A common test thread pool suitable for reasonably low concurrency tests.
   */
  public static final Executor threadPool = Executors.newFixedThreadPool(4);

  public static int getSerializedSize(final Serializable object) throws IOException {
    final ByteArrayOutputStream bout = new ByteArrayOutputStream();
    try (final ObjectOutputStream oout = new ObjectOutputStream(bout)) {
      oout.writeObject(object);
    }
    return bout.size();
  }

  public static <T extends Serializable> T serialize(final T object) throws IOException, ClassNotFoundException {
    final ByteArrayOutputStream bout = new ByteArrayOutputStream();
    try (final ObjectOutputStream oout = new ObjectOutputStream(bout)) {
      oout.writeObject(object);
    }

    final ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
    try (final ObjectInputStream oin = new ObjectInputStream(bin)) {
      return (T) oin.readObject();
    }
  }

  @FunctionalInterface
  public static interface CheckedRunnable {
    void run() throws Exception;

    @SneakyThrows
    default void runUnchecked() {
      run();
    }
  }

  /**
   * Wraps a checked lambda in {@link SneakyThrows}.
   */
  public static Runnable unchecked(final CheckedRunnable checked) {
    return () -> checked.runUnchecked();
  }

  public static ActionCluster.Node testNode(final ActionCluster cluster, final EmissionMonitor<Long> monitor) {
    return cluster.new Node(() -> monitor.emit(Scheduler.global.now()));
  }

  /**
   * Activates a posterior via its integrator. Useful for tests that rely on
   * integrator states. By itself, this will cause the posterior to activate after
   * {@code IntegrationProfile.TRANSIENT.defaultInterval()}.
   */
  public static void triggerPosterior(final Posterior node) {
    node.getIntegrator().add(IntegrationProfile.TRANSIENT, Prior.DEFAULT_COEFFICIENT);
  }

  /**
   * Inhibits a posterior. Useful for disassociation tests.
   */
  public static void inhibitPosterior(final Posterior node) {
    node.getIntegrator().add(IntegrationProfile.TRANSIENT, -Prior.DEFAULT_COEFFICIENT);
  }
}
