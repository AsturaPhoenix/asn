package ai.xng;

import java.util.Optional;

import ai.xng.BakingIntegrator.Segment;
import ai.xng.util.TimeSeries;
import io.reactivex.disposables.Disposable;
import lombok.val;

public abstract class ThresholdIntegrator {
  public static final float THRESHOLD = 1;

  public class Spike {
    private final Segment rampUp, rampDown;

    private Spike(final IntegrationProfile profile, final float magnitude) {
      final long now = Scheduler.global.now();
      rampUp = new Segment(now + profile.delay(), now + profile.peak(), 0, magnitude / profile.rampUp());
      rampDown = new Segment(rampUp.t1, now + profile.period(), magnitude, -magnitude / profile.rampDown());
    }

    /**
     * Modifies the spike to use a new ramp-up rate. If a ramp-up is already in
     * progress, the ramp-up segment is moved to begin at its current coordinate.
     * There is no effect if the peak has already passed.
     */
    public void adjustRampUp(final float newRate) {
      final long now = Scheduler.global.now();
      if (now < rampUp.t1) {
        if (now > rampUp.t0) {
          rampUp.v0 = rampUp.evaluate(now);
        }
        rampUp.rate = newRate;
        rampDown.v0 = rampUp.evaluate(rampUp.t1);
        rampDown.rate = -rampDown.v0 / rampDown.duration();
        invalidate();
      }
    }

    public long end() {
      return rampDown.t1;
    }

    public void clear() {
      integrator.remove(rampUp);
      integrator.remove(rampDown);
      invalidate();
    }
  }

  private final BakingIntegrator integrator = new BakingIntegrator();

  public boolean isEmpty() {
    return integrator.isEmpty();
  }

  private TimeSeries<Disposable> nextThreshold;

  /**
   * Gets the timestamp of the currently scheduled next threshold.
   */
  public Optional<Long> nextThreshold() {
    return Optional.ofNullable(nextThreshold).map(TimeSeries::time);
  }

  /**
   * Returns whether there is currently a due threshold processing task scheduled.
   */
  public boolean isPending() {
    return nextThreshold().map(t -> t <= Scheduler.global.now()).orElse(false);
  }

  public float getValue() {
    return getValue(Scheduler.global.now());
  }

  public float getValue(final long t) {
    return integrator.evaluate(t).value();
  }

  public float getNormalizedCappedValue() {
    val value = getValue();
    return value >= THRESHOLD ? 1 : value / THRESHOLD;
  }

  protected abstract void onThreshold();

  public Spike add(final IntegrationProfile profile, final float magnitude) {
    evict();
    val spike = new Spike(profile, magnitude);
    integrator.add(spike.rampUp);
    integrator.add(spike.rampDown);
    invalidate();
    return spike;
  }

  private void invalidate() {
    val now = Scheduler.global.now();
    if (nextThreshold != null && nextThreshold.time() == now) {
      return;
    }

    final Optional<Long> updatedNextThreshold = nextThreshold(now);
    if (nextThreshold == null || updatedNextThreshold.map(t -> t != nextThreshold.time())
        .orElse(true)) {
      if (nextThreshold != null) {
        nextThreshold.value().dispose();
      }
      schedule(updatedNextThreshold);
    }
  }

  public Optional<Long> nextThreshold(long t) {
    while (true) {
      val trajectory = integrator.evaluate(t);
      val nextCriticalPoint = integrator.nextCriticalPoint(t);

      if (trajectory.value() < THRESHOLD) {
        if (trajectory.rate() > 0) {
          final long intercept = t + (long) Math.ceil((THRESHOLD - trajectory.value()) / trajectory.rate());
          if (nextCriticalPoint.map(tc -> intercept <= tc)
              .orElse(true)) {
            return Optional.of(intercept);
          }
        }
      }

      if (nextCriticalPoint.isEmpty())
        break;
      t = nextCriticalPoint.get();
    }

    return Optional.empty();
  }

  public boolean isActive() {
    return getValue() >= THRESHOLD;
  }

  public void evict() {
    integrator.evict(Scheduler.global.now());
  }

  private void schedule(final Optional<Long> tOpt) {
    nextThreshold = tOpt.map(t -> new TimeSeries<Disposable>(Scheduler.global.postTask(() -> {
      onThreshold();
      evict();
      schedule(nextThreshold(Scheduler.global.now()));
    }, t), t))
        .orElse(null);
  }

  @Override
  public String toString() {
    return integrator.evaluate(Scheduler.global.now()) + ": " + integrator.toString();
  }
}