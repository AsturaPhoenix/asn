package ai.xng;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import lombok.AllArgsConstructor;
import lombok.val;

public class BakingIntegrator {
  @AllArgsConstructor
  public static class Segment {
    public long t0, t1;
    public float v0, rate;

    public float evaluate(final long t) {
      return v0 + rate * (t - t0);
    }

    public long duration() {
      return t1 - t0;
    }
  }

  public static record Ray(float value, float rate) {
  }

  private final List<Segment> segments = new ArrayList<>();

  public boolean isEmpty() {
    return segments.isEmpty();
  }

  public void add(final Segment segment) {
    segments.add(segment);
  }

  public void remove(final Segment segment) {
    segments.remove(segment);
  }

  public void evict(final long t) {
    segments.removeIf(s -> s.t1 <= t);
  }

  public Ray evaluate(final long t) {
    float vt = 0, rate = 0;
    for (val s : segments) {
      if (s.t0 <= t && t < s.t1) {
        vt += s.evaluate(t);
        rate += s.rate;
      }
    }
    return new Ray(vt, rate);
  }

  public Optional<Long> nextCriticalPoint(final long t) {
    Optional<Long> next = Optional.empty();
    for (val s : segments) {
      Optional<Long> candidate = Optional.empty();
      if (s.t0 > t) {
        candidate = Optional.of(s.t0);
      } else if (s.t1 > t) {
        candidate = Optional.of(s.t1);
      }
      if (candidate.isPresent() && (!next.isPresent() || candidate.get() < next.get())) {
        next = candidate;
      }
    }
    return next;
  }

  @Override
  public String toString() {
    return segments.toString();
  }
}