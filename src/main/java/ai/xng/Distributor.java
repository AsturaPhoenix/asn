package ai.xng;

import java.util.ArrayList;
import java.util.Collection;
import java.util.function.Consumer;

import lombok.val;

/**
 * Executes actions distributed over a set of items by weight. Plausible
 * implementations proportional, inverse proprotional (harmonic), and softmin.
 */
public interface Distributor {
  /**
   * Adds an item to this distributor. The action will be called with the
   * finalized adjusted proportion.
   */
  void add(final float weight, final Consumer<Float> action);

  /**
   * Calculates adjusted proportions for each item added and calls the action for
   * all nonzero proportions.
   */
  void distribute();

  void clear();

  class Inverse implements Distributor {
    private static record Item(float inverseWeight, Consumer<Float> action) {
    }

    private float total;
    private final Collection<Item> items = new ArrayList<>();

    private boolean isDegenerate() {
      return total < 0;
    }

    private void setDegenerate() {
      total = -1;
    }

    public void clear() {
      total = 0;
      items.clear();
    }

    @Override
    public void add(final float weight, final Consumer<Float> action) {
      if (weight == 0) {
        if (!isDegenerate()) {
          setDegenerate();
          items.clear();
        }
        items.add(new Item(0, action));
      } else {
        if (!isDegenerate()) {
          final float inverseWeight = 1 / weight;
          total += inverseWeight;
          items.add(new Item(inverseWeight, action));
        }
      }
    }

    @Override
    public void distribute() {
      if (isDegenerate()) {
        final float proportion = 1 / items.size();
        for (val item : items) {
          item.action.accept(proportion);
        }
      } else {
        for (val item : items) {
          item.action.accept(item.inverseWeight / total);
        }
      }
    }
  }
}
