package ai.xng.constructs;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import ai.xng.ActionCluster;
import ai.xng.DataCluster;
import ai.xng.InputCluster;
import ai.xng.Node;
import lombok.val;

public class CharacterDecoder implements Serializable {
  private static interface Predicate extends Serializable {
    boolean apply(final int codePoint);
  }

  private static record Test(Predicate predicate, InputCluster.Node output) implements Serializable {
  }

  public final InputCluster.Node isWhitespace,
      isJavaIdentifierStart,
      isJavaIdentifierPart,
      isDigit,
      isUpperCase,
      isLowerCase,
      isControl,
      isOther;

  private final Test[] tests;

  private final InputCluster output;
  private final Map<Integer, InputCluster.Node> oneHotEncoding = new HashMap<>();

  public final ActionCluster.Node node;

  public CharacterDecoder(final ActionCluster actionCluster, final DataCluster input, final InputCluster output) {
    this.output = output;
    tests = new Test[] {
        new Test(Character::isWhitespace, isWhitespace = output.new Node() {
          @Override
          public String toString() {
            return "isWhitespace";
          }
        }),
        new Test(Character::isJavaIdentifierStart, isJavaIdentifierStart = output.new Node() {
          @Override
          public String toString() {
            return "isIdentifierStart";
          }
        }),
        new Test(Character::isJavaIdentifierPart, isJavaIdentifierPart = output.new Node() {
          @Override
          public String toString() {
            return "isIdentifierPart";
          }
        }),
        new Test(Character::isDigit, isDigit = output.new Node() {
          @Override
          public String toString() {
            return "isDigit";
          }
        }),
        new Test(Character::isUpperCase, isUpperCase = output.new Node() {
          @Override
          public String toString() {
            return "isUpperCase";
          }
        }),
        new Test(Character::isLowerCase, isLowerCase = output.new Node() {
          @Override
          public String toString() {
            return "isLowerCase";
          }
        }),
        new Test(Character::isISOControl, isControl = output.new Node() {
          @Override
          public String toString() {
            return "isControl";
          }
        })
    };

    isOther = output.new Node() {
      @Override
      public String toString() {
        return "unknown character class";
      }
    };

    node = new CoincidentEffect.Lambda<>(actionCluster, input, node -> {
      if (node.getData()instanceof Integer codePoint) {
        forOutput(codePoint, Node::activate);
      }
    }).node;
  }

  public void forOutput(final int codePoint, final Consumer<? super InputCluster.Node> action) {
    boolean isClassified = false;
    for (val test : tests) {
      if (test.predicate().apply(codePoint)) {
        action.accept(test.output());
        isClassified = true;
      }
    }
    if (!isClassified) {
      action.accept(isOther);
    }

    action.accept(oneHotEncoding.computeIfAbsent(Character.toLowerCase(codePoint), __ -> output.new Node() {
      @Override
      public String toString() {
        return new StringBuilder("'").appendCodePoint(codePoint).append('\'').toString();
      }
    }));
  }
}
