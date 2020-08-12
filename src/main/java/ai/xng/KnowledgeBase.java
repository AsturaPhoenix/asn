package ai.xng;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.MapMaker;

import ai.xng.NGramWindow.SymbolPair;
import io.reactivex.Completable;
import io.reactivex.Observable;
import io.reactivex.Observer;
import io.reactivex.disposables.Disposable;
import io.reactivex.subjects.CompletableSubject;
import io.reactivex.subjects.PublishSubject;
import io.reactivex.subjects.Subject;
import lombok.val;

/**
 * Since this is meant to be a proof of concept, encapsulation is not perfect.
 * Particularly, the following external operations should be avoided:
 * 
 * <ul>
 * <li>{@link Node#setOnActivate(java.util.function.Consumer)}
 * <li>mutators on {@link Context#index}
 * </ul>
 */
public class KnowledgeBase implements Serializable, AutoCloseable {
  private static final long serialVersionUID = 5249665334141533302L;

  public static final int DEFAULT_MAX_STACK_DEPTH = 512, DEFAULT_LOOKBEHIND = 5, DEFAULT_LOOKAHEAD = 2;

  /**
   * Map that keeps its nodes alive. Note that the {@code weakKeys} configuration
   * here is just to use identity keys with concurrency; since nodes have strong
   * references to their values, they will never actually be evicted.
   */
  private transient Map<Object, SynapticNode> strongIndex;
  /**
   * Map that does not keep its nodes alive. This is suitable only for values that
   * refer to their nodes, like {@link Context}s.
   */
  private transient Map<Object, SynapticNode> weakIndex;

  private transient ThreadPoolExecutor threadPool;
  private transient Subject<String> rxOutput;

  private static String toQualifiedString(final Enum<?> e) {
    return String.format("%s.%s", e.getDeclaringClass()
        .getSimpleName(), e.name());
  }

  public enum Common {
    // invocation
    literal, transform, exceptionHandler,
    // If present as a property on an invocation node, the value node is activated
    // after
    // the child context goes inactive, after which all return values (if any) are
    // unpacked into an ordinal array typed by the value node.
    afterAll,
    // If present as a property on an invocation node, the value node is activated
    // in a tail recursion context for every return value of the invoked function.
    forEach,

    // When a value is assigned to this key, it is also assigned to the parent
    // context keyed by the invocation node.
    returnValue,

    // stack frames
    caller, invocation, context, maxStackDepth,

    // ordinals
    type, ordinal,

    // parsing
    rewriteWindow, symbol, codePoint, lookahead, lookbehind, rewriteLength,

    javaClass, argument, parameter, object, name, exception, source, value, entrypoint, relative,

    // association
    prior, posterior, coefficient,

    /**
     * Used as a value node for the {@link BuiltIn#eavTuple} built-in to indicate
     * the EAV node triggered when a property is unset. This should not be used for
     * {@link KnowledgeBase#eavNode(Node...)}, where {@code null} should be used
     * directly instead.
     */
    eavUnset;

    @Override
    public String toString() {
      return toQualifiedString(this);
    }
  }

  public enum BuiltIn {
    /**
     * Activates the node at {@link Common#value} in the context given by
     * {@link Common#context}, or the calling context if absent. This activation
     * does not block.
     * 
     * As an alternative, consider chaining off of EAV nodes instead of activating
     * contextual nodes. However, this is useful for function pointers as it can
     * invoke an entrypoint with indirection.
     */
    activate {
      @Override
      public Node impl(final KnowledgeBase kb, final Context context) {
        kb.requireNode(context, Common.value)
            .activate(kb.optional(context, Common.context, () -> kb.getParent(context)));
        return null;
      }
    },
    associate {
      @Override
      public Node impl(final KnowledgeBase kb, final Context context) {
        final Node prior = kb.requireNode(context, Common.prior);
        final SynapticNode posterior = (SynapticNode) kb.requireNode(context, Common.posterior);
        final float coefficient = kb.optional(context, Common.coefficient, Node.DEFAULT_COEFFICIENT);

        posterior.getSynapse()
            .setCoefficient(prior, coefficient);
        return null;
      }
    },
    /**
     * <ul>
     * <li>{@link Common#value} - required; string to rewrite
     * <li>{@link Common#lookbehind} - optional; number of symbols behind cursor to
     * include in window; defaults to {@link #DEFAULT_LOOKBEHIND}
     * <li>{@link Common#lookahead} - optional; number of symbols ahead of cursor to
     * include in window; defaults to {@link #DEFAULT_LOOKAHEAD}
     * </ul>
     */
    createRewriteWindow {
      @Override
      protected Node impl(final KnowledgeBase kb, final Context context) throws Exception {
        return kb.node(new NGramWindow(
            kb.<String>require(context, Common.value)
                .codePoints()
                .boxed()
                .map(codePoint -> new SymbolPair(kb.node(Common.codePoint), kb.node(codePoint))),
            kb.optional(context, Common.lookbehind, DEFAULT_LOOKBEHIND),
            kb.optional(context, Common.lookahead, DEFAULT_LOOKAHEAD)));
      }
    },
    /**
     * Activates EAV nodes for symbols in a rewrite window. The keys to the EAV
     * nodes are symbol ordinals, where the 0 position is the first symbol behind
     * the current position. The EAV nodes are activated in the parent context.
     * 
     * <ul>
     * <li>{@link Common#rewriteWindow} - required
     * </ul>
     */
    dispatchNGrams {
      @Override
      protected Node impl(final KnowledgeBase kb, final Context context) {
        final NGramWindow rewriter = kb.require(context, Common.rewriteWindow);

        // n-gram ordinals range from -lookbehind + 1 to lookahead, where 0 is the most
        // recent 1-gram.
        final int ordinalShift = 1 - Math.min(rewriter.position, rewriter.config.behind());

        kb.unpackList(rewriter.window(), kb.getParent(context), ImmutableMap.of(
            kb.node(Common.symbol), SymbolPair::symbol,
            kb.node(Common.value), SymbolPair::value), ordinalShift);
        return null;
      }
    },
    /**
     * Produces a rewrite window based on the window at
     * {@link Common#rewriteWindow}.
     */
    rewrite {
      @Override
      protected Node impl(final KnowledgeBase kb, final Context context) {
        final NGramWindow rewriter = kb.require(context, Common.rewriteWindow);
        final int rewriteLength = kb.require(context, Common.rewriteLength);
        val replacement = kb.packList(context, SymbolPair::new, Common.symbol, Common.value);
        return kb.node(rewriter.rewrite(rewriteLength, replacement));
      }
    },
    /**
     * Get the node representing an entity-attribute-value tuple. This node is
     * automatically activated when a matching property is queried or written.
     * 
     * If only one of {@link Common#object} or {@link Common#name} is specified, a
     * relative EAV node to the contextual EAV node for the specified node is
     * returned. Alternatively, {@link Common#relative} is a boolean to indicate a
     * fully specified relative EAV node.
     * 
     * If {@link Common#value} is omitted, the wildcard EAV node is returned. To
     * retrieve the EAV node for the clearing of a property, pass
     * {@link Common#eavUnset}.
     */
    eavNode {
      @Override
      public Node impl(final KnowledgeBase kb, final Context context) {
        final Node object = kb.optionalNode(context, Common.object),
            name = kb.optionalNode(context, Common.name),
            value = kb.optionalNode(context, Common.value);

        val path = new ArrayList<Node>(3);
        if (object != null)
          path.add(object);
        if (name != null)
          path.add(name);
        final boolean relative = path.size() == 1 || kb.<Boolean>require(context, Common.relative);

        if (value != null)
          path.add(value.getValue() == Common.eavUnset ? null : value);

        return kb.eavNode(relative, value == null, path.toArray(new Node[] {}));
      }
    },
    /**
     * Gets a static or instance Java field.
     */
    field {
      @Override
      public Node impl(final KnowledgeBase kb, final Context context) throws Exception {
        final Object object = kb.optional(context, Common.object, null);
        final Class<?> clazz = kb.optional(context, Common.javaClass, object::getClass);
        final String name = kb.require(context, Common.name);

        return kb.node(clazz.getField(name)
            .get(object));
      }
    },
    /**
     * Takes two args: {@link Common#object} and {@link Common#name}. If
     * {@code object} is omitted, gets the property from the parent context.
     * 
     * Additionally, invokes AV/EAV nodes for the retrieved property in the parent
     * context.
     */
    getProperty {
      @Override
      public Node impl(final KnowledgeBase kb, final Context context) {
        final Node object = kb.optionalNode(context, Common.object),
            property = kb.requireNode(context, Common.name);
        final Context parent = kb.getParent(context);
        final Node value = (object == null ? parent.node : object).properties.get(property);
        kb.activateEavNodes(context, object, property, value);
        return value;
      }
    },
    /**
     * A limited Java interop bridge. Method parameter types must be fully and
     * exactly specified via "paramN" index entries, null-terminated, having values
     * of Java classes. Arguments are passed as "argN" index entries. Missing
     * arguments are passed null.
     * <p>
     * If the class is not explicitly specified, this bridge may attempt to use the
     * inaccessible concrete class of the object. More mature knowledge bases should
     * implement a more intelligent interop mechanism, which may include better
     * resolution, caching, and/or lambda metafactory.
     * 
     * {@link Common#javaClass}: optional class on which to invoke method. If
     * {@link Common#object} is also provided, this must be a supertype.
     * {@link Common#object}: optional object on which to invoke method.
     * {@link Common#name}: required name of the method to invoke.
     */
    method {
      @Override
      public Node impl(final KnowledgeBase kb, final Context context) throws Exception {
        final Object object = kb.optional(context, Common.object, null);
        final Class<?> clazz = kb.optional(context, Common.javaClass, object::getClass);
        final String name = kb.require(context, Common.name);

        val args = kb.packList(context, (type, value) -> new Argument(
            (Class<?>) type.getValue(),
            value != null ? value.getValue() : null),
            Common.parameter, Common.argument);

        final Method method = clazz.getMethod(name, args.stream()
            .map(Argument::type)
            .toArray(Class<?>[]::new));
        final Object ret = method.invoke(object, args.stream()
            .map(Argument::value)
            .toArray());
        return method.getReturnType() == null ? null : kb.node(ret);
      }
    },
    ordinal {
      @Override
      protected Node impl(final KnowledgeBase kb, final Context context) throws Exception {
        return kb.node(new Ordinal(kb.requireNode(context, Common.type), kb.require(context, Common.ordinal)));
      }
    },
    findClass {
      @Override
      public Node impl(final KnowledgeBase kb, final Context context) throws ClassNotFoundException {
        final Class<?> clazz;
        final String name = kb.require(context, Common.name);
        switch (name) {
        case "void":
          clazz = void.class;
          break;
        case "boolean":
          clazz = boolean.class;
          break;
        case "byte":
          clazz = byte.class;
          break;
        case "char":
          clazz = char.class;
          break;
        case "short":
          clazz = short.class;
          break;
        case "int":
          clazz = int.class;
          break;
        case "long":
          clazz = long.class;
          break;
        case "float":
          clazz = float.class;
          break;
        case "double":
          clazz = double.class;
          break;
        default:
          clazz = Class.forName(name);
        }
        return kb.node(clazz);
      }
    },
    /**
     * Creates a new anonymous node. Invocation nodes are created by specifying a
     * {@link Common#entrypoint}, which serves as the entrypoint.
     */
    node {
      @Override
      public Node impl(final KnowledgeBase kb, final Context context) {
        final Node entrypoint = kb.optionalNode(context, Common.entrypoint);
        return entrypoint == null ? new SynapticNode() : kb.new InvocationNode(entrypoint);
      }
    },
    print {
      @Override
      public Node impl(final KnowledgeBase kb, final Context context) {
        kb.rxOutput.onNext(Objects.toString(kb.require(context, Common.value)));
        return null;
      }
    },
    /**
     * <li>{@link Common#object}: optional node on which to set a property. If
     * omitted, sets the property on the parent context.
     * <li>{@link Common#name}: required property name.
     * <li>{@link Common#value}: optional value to set. If null, the property is
     * cleared.
     * <li>Returns the value previously at this property.
     * 
     * Additionally, invokes AV/EAV nodes for the new property in the parent
     * context.
     * 
     * The special context key {@link Common#returnValue} is mirrored to the parent
     * context keyed by the invocation node, and may trigger the continuation of the
     * invocation activation chain.
     */
    setProperty {
      @Override
      public Node impl(final KnowledgeBase kb, final Context context) {
        return kb.setProperty(kb.getParent(context), kb.optionalNode(context, Common.object),
            kb.requireNode(context, Common.name), kb.optionalNode(context, Common.value));
      }
    };

    private static record Argument(Class<?> type, Object value) {
    }

    protected abstract Node impl(KnowledgeBase kb, Context context) throws Exception;

    private SynapticNode node(final KnowledgeBase kb) {
      return new SynapticNode(this) {
        private static final long serialVersionUID = -307360749192088062L;

        @Override
        protected Completable onActivate(final Context context) {
          return Completable
              .fromAction(() -> kb.setProperty(context, null, kb.node(Common.returnValue), impl(kb, context)));
        }
      };
    }

    @Override
    public String toString() {
      return toQualifiedString(this);
    }
  }

  public Node requireNode(final Context context, final Common parameter) {
    return context.require(node(parameter));
  }

  @SuppressWarnings("unchecked")
  public <T> T require(final Context context, final Common parameter) {
    return (T) requireNode(context, parameter).getValue();
  }

  public Node optionalNode(final Context context, final Common parameter) {
    return context.node.properties.get(node(parameter));
  }

  @SuppressWarnings("unchecked")
  public <T> T optional(final Context context, final Common parameter, final Supplier<T> computeDefault) {
    final Node node = optionalNode(context, parameter);
    return node != null ? (T) node.getValue() : computeDefault.get();
  }

  public <T> T optional(final Context context, final Common parameter, final T defaultValue) {
    return optional(context, parameter, () -> defaultValue);
  }

  public <T> void unpackList(final List<T> list, final Context context, final Map<Node, Function<T, Node>> extractors,
      final int ordinalShift) {
    for (int i = 0; i < list.size(); ++i) {
      for (val extractor : extractors.entrySet()) {
        setProperty(context, null, node(new Ordinal(extractor.getKey(), i + ordinalShift)), extractor.getValue()
            .apply(list.get(i)));
      }
    }
  }

  public void unpackList(final List<Node> list, final Context context, final Node ordinalType) {
    unpackList(list, context, ImmutableMap.of(ordinalType, Function.identity()), 0);
  }

  public <T> ImmutableList<T> packList(final Context context, final Function<Node[], T> aggregator,
      Common... ordinalTypes) {
    ImmutableList.Builder<T> builder = ImmutableList.builder();
    val terms = new Node[ordinalTypes.length];

    {
      boolean hasSingle = false;
      for (int j = 0; j < ordinalTypes.length; ++j) {
        val term = optionalNode(context, ordinalTypes[j]);
        terms[j] = term;
        if (term != null) {
          hasSingle = true;
        }
      }
      if (hasSingle) {
        builder.add(aggregator.apply(terms));
        return builder.build();
      }
    }

    for (int i = 0;; ++i) {
      boolean hasElement = false;
      for (int j = 0; j < ordinalTypes.length; ++j) {
        val term = context.node.properties.get(node(new Ordinal(node(ordinalTypes[j]), i)));
        terms[j] = term;
        if (term != null) {
          hasElement = true;
        }
      }

      if (hasElement) {
        builder.add(aggregator.apply(terms));
      } else {
        break;
      }
    }

    return builder.build();
  }

  public <T> ImmutableList<T> packList(final Context context, final BiFunction<Node, Node, T> aggregator,
      Common type1, Common type2) {
    return packList(context, e -> aggregator.apply(e[0], e[1]), type1, type2);
  }

  public KnowledgeBase() {
    preInit();
    postInit();
    bootstrap();
  }

  private void preInit() {
    strongIndex = new MapMaker().weakKeys()
        .makeMap();
    weakIndex = new MapMaker().weakKeys()
        .weakValues()
        .makeMap();

    preInitEav();

    rxOutput = PublishSubject.create();

    final int corePoolSize = Runtime.getRuntime()
        .availableProcessors();
    threadPool = new ThreadPoolExecutor(corePoolSize, 4 * corePoolSize, 60, TimeUnit.SECONDS, new SynchronousQueue<>());
  }

  private void postInit() {
    postInitEav();
  }

  private static class KbContext extends ai.xng.Context {
    static final long serialVersionUID = 4387542426477791967L;

    List<Context> children = new ArrayList<>();

    KbContext(final KnowledgeBase kb) {
      super(kb::node, () -> kb.threadPool);
    }

    CompletableFuture<Void> reinforceRecursively(final Optional<Long> decayPeriod, final float weight) {
      val async = new AsyncJoiner();

      getScheduler().ensureOnThread(() -> {
        for (final Context child : children) {
          async.add(child.reinforce(decayPeriod, weight));
        }
        async.arrive();
      });

      return async.future;
    }

    void shutDownRecurisvely() {
      getScheduler().shutdown();
      getScheduler().pause();
      for (final Context child : children) {
        shutDownRecursively(child);
      }
    }
  }

  /**
   * Create a context bound to this knowledge base, which registers its node
   * representation and uses a common thread pool.
   */
  public Context newContext() {
    return new KbContext(this);
  }

  private static Node propertyDescent(Node node, final Node... properties) {
    for (final Node property : properties) {
      if (node == null)
        return null;
      node = node.properties.get(property);
    }
    return node;
  }

  private Node propertyDescent(final Node node, final Common... properties) {
    final Node[] propertyNodes = new Node[properties.length];
    for (int i = 0; i < properties.length; ++i) {
      propertyNodes[i] = node(properties[i]);
    }
    return propertyDescent(node, propertyNodes);
  }

  /**
   * Attempts to get the parent context of a given context. If the context does
   * not have a parent, the original context is used instead. If the parent node
   * is not a context, {@link ClassCastException} is thrown.
   * 
   * There is also an edge case where the parent node has a null value, in which
   * case null is returned.
   */
  private Context getParent(final Context context) {
    final Node parentNode = propertyDescent(context.node, Common.caller, Common.context);
    return parentNode != null ? (Context) parentNode.getValue() : context;
  }

  private void inheritActivity(final Context parent, final Context child) {
    // Tie the parent activity to the child activity.
    child.rxActive()
        .subscribe(new Observer<Boolean>() {
          Context.Ref ref;

          @Override
          public void onNext(Boolean active) {
            if (active) {
              assert ref == null;
              ref = parent.new Ref();
            } else if (!active && ref != null) {
              ref.close();
              ref = null;
            }
          }

          @Override
          public void onSubscribe(Disposable d) {
          }

          @Override
          public void onError(Throwable e) {
            e.printStackTrace();
            if (ref != null)
              ref.close();
          }

          @Override
          public void onComplete() {
            if (ref != null)
              ref.close();
          }
        });
  }

  public static record NodeStackFrame(InvocationNode invocation, Context context) implements Serializable {
    private static final long serialVersionUID = 1L;

    @Override
    public String toString() {
      val sb = new StringBuilder("from ").append(invocation.debugDump(SKIP_CALLER));
      for (val traceElement : invocation.trace) {
        sb.append("\n        at ")
            .append(traceElement);
      }
      return sb.toString();
    }

    public Stream<NodeStackFrame> trace(final KnowledgeBase kb) {
      final Node callerKey = kb.node(Common.caller), invocationKey = kb.node(Common.invocation),
          contextKey = kb.node(Common.context);

      return Stream.iterate(this, s -> {
        final Node caller = s.context.node.properties.get(callerKey);
        if (caller == null)
          return null;

        final Node invocation = caller.properties.get(invocationKey), contextNode = caller.properties.get(contextKey);
        final Object context = contextNode == null ? null : contextNode.getValue();

        return invocation instanceof InvocationNode i && context instanceof Context c ? new NodeStackFrame(i, c) : null;
      })
          .takeWhile(s -> s != null);
    }

    public String traceAsString(final KnowledgeBase kb) {
      return trace(kb).map(Object::toString)
          .collect(Collectors.joining("\n"));
    }
  }

  /**
   * Exception thrown when a {@link KnowledgeBase} invocation exceeds the stack
   * depth limit specified in the context (as
   * {@link KnowledgeBase.Common#maxStackDepth}).
   */
  public class StackDepthExceededException extends InvocationException {
    private static final long serialVersionUID = 1L;

    private StackDepthExceededException(final NodeStackFrame stackFrame) {
      super("Recursion limit exceeded.", stackFrame);
    }
  }

  public class InvocationException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public final NodeStackFrame stackFrame;

    private InvocationException(final Throwable cause, final NodeStackFrame stackFrame) {
      super(stackFrame.traceAsString(KnowledgeBase.this), cause);
      this.stackFrame = stackFrame;
    }

    private InvocationException(final String message, final NodeStackFrame stackFrame) {
      super(String.format("%s\n%s", message, stackFrame.traceAsString(KnowledgeBase.this)));
      this.stackFrame = stackFrame;
    }
  }

  /**
   * Invocation nodes activate the specified entry point node with a child context
   * constructed from their {@link Common#literal} and {@link Common#transform}
   * properties. Literal properties are assigned directly to the new context, and
   * transform properties are copied from the parent context entry named by the
   * value of the transform property. Transform properties take precedence over
   * literal properties if present, allowing literals to act as defaults. The
   * return value, if any, is assigned to the context property named by the node
   * with the invocation, whose activation triggered the invocation.
   * 
   * Tail recursion can be achieved by setting the caller to the parent caller
   * (via {@code transform} or {@link #inherit(Node)}).
   */
  public class InvocationNode extends SynapticNode {
    private static final long serialVersionUID = 3233708822279852070L;

    // This is only for debug and will get less useful as the knowledge base
    // evolves.
    private final List<StackTraceElement> trace;

    public InvocationNode(final Node invoke) {
      super(invoke);
      val fullTrace = Arrays.asList(Thread.currentThread()
          .getStackTrace());
      trace = ImmutableList.copyOf(fullTrace.subList(2, fullTrace.size()));
    }

    public InvocationNode(final Serializable invoke) {
      this(node(invoke));
    }

    @Override
    public String toString() {
      val sb = new StringBuilder(Integer.toHexString(hashCode()));
      if (comment != null) {
        sb.append(": ")
            .append(comment);
      }
      sb.append(" = Invocation of ")
          .append(getValue());
      return sb.toString();
    }

    @Override
    protected Completable onActivate(final Context parentContext) {
      val maxStackDepthNode = parentContext.node.properties.get(node(Common.maxStackDepth));
      final int maxStackDepth = maxStackDepthNode == null || !(maxStackDepthNode.getValue() instanceof Integer)
          ? DEFAULT_MAX_STACK_DEPTH
          : (int) maxStackDepthNode.getValue();
      if (maxStackDepth == 0)
        return Completable.error(new StackDepthExceededException(new NodeStackFrame(this, parentContext)));

      final Context childContext = newContext();

      final Node afterAll = properties.get(node(Common.afterAll));
      if (afterAll != null) {
        parentContext.registerReturnValueCollector(afterAll);
      }

      // Hold a ref while we're starting the invocation to avoid pulsing the EAV nodes
      // and making it look like we're transitioning to idle early.
      try (val ref = childContext.new Ref()) {
        if (parentContext instanceof KbContext kbContext) {
          kbContext.children.add(childContext);
        }

        final Node caller = new SynapticNode();
        setProperty(childContext, null, node(Common.caller), caller);
        setProperty(childContext, caller, node(Common.invocation), this);
        setProperty(childContext, caller, node(Common.context), parentContext.node);

        setProperty(childContext, null, node(Common.maxStackDepth), node(maxStackDepth - 1));
        inheritActivity(parentContext, childContext);

        final Node literal = properties.get(node(Common.literal));
        if (literal != null) {
          synchronized (literal.properties) {
            for (final Entry<Node, Node> mapping : literal.properties.entrySet()) {
              setProperty(childContext, null, mapping.getKey(), mapping.getValue());
            }
          }
        }

        final Node transform = properties.get(node(Common.transform));
        if (transform != null) {
          synchronized (transform.properties) {
            for (final Entry<Node, Node> mapping : transform.properties.entrySet()) {
              final Node value = parentContext.node.properties.get(mapping.getValue());

              // Note that this implies that missing transform values will not have their
              // null-value EAV nodes pulsed, but nor will they have their EAV set nodes
              // pulsed. While different from the behavior for literals, this is akin to the
              // argument not being passed in, and is more consistent for tail recursion.
              if (value != null) {
                setProperty(childContext, null, mapping.getKey(), value);
              }
            }
          }
        }

        final Node exceptionHandler = properties.get(node(Common.exceptionHandler));
        if (exceptionHandler == null) {
          childContext.exceptionHandler = e -> childContext.continuation()
              .completeExceptionally(e instanceof InvocationException ? e
                  : new InvocationException(e, new NodeStackFrame(this, parentContext)));
        } else {
          childContext.exceptionHandler = e -> {
            final Node exceptionNode = node(new ImmutableException(e));
            exceptionNode.properties.put(node(Common.source), this);
            // Right now we use the same context for the exception handler, so if another
            // exception is thrown the exception handler is invoked again (and is likely to
            // hit a refractory dedup). We should consider changing this once contexts can
            // inherit activations.
            setProperty(childContext, null, node(Common.exception), exceptionNode);
            exceptionHandler.activate(childContext);
          };
        }

        ((Node) getValue()).activate(childContext);
      }

      val completable = CompletableSubject.create();
      childContext.continuation()
          .thenRun(completable::onComplete)
          .exceptionally(t -> {
            completable.onError(t instanceof CompletionException ? t.getCause() : t);
            return null;
          });
      if (afterAll != null) {
        childContext.rxActive()
            .filter(active -> !active)
            .firstElement()
            .ignoreElement()
            .observeOn(parentContext.getScheduler())
            .subscribe(() -> {
              val returnValues = parentContext.getReturnValues(afterAll);
              if (returnValues != null) {
                unpackList(returnValues, parentContext, afterAll);
              }
              afterAll.activate(parentContext);
            });
      }
      return completable;
    }

    public InvocationNode literal(final Node key, final Node value) {
      if (key == null) {
        throw new NullPointerException("Key cannot be null.");
      }

      val literals = properties.computeIfAbsent(node(Common.literal), k -> new SynapticNode()).properties;
      if (value == null) {
        literals.remove(key);
      } else {
        literals.put(key, value);
      }
      return this;
    }

    public InvocationNode literal(final Serializable key, final Node value) {
      return literal(node(key), value);
    }

    public InvocationNode literal(final Node key, final Serializable value) {
      return literal(key, node(value));
    }

    public InvocationNode literal(final Serializable key, final Serializable value) {
      return literal(node(key), node(value));
    }

    public InvocationNode transform(final Node key, final Node value) {
      if (key == null) {
        throw new NullPointerException("Key cannot be null.");
      }

      val transforms = properties.computeIfAbsent(node(Common.transform), k -> new SynapticNode()).properties;
      if (value == null) {
        transforms.remove(key);
      } else {
        transforms.put(key, value);
      }
      return this;
    }

    public InvocationNode transform(final Serializable key, final Node value) {
      return transform(node(key), value);
    }

    public InvocationNode transform(final Node key, final Serializable value) {
      return transform(key, node(value));
    }

    public InvocationNode transform(final Serializable key, final Serializable value) {
      return transform(node(key), node(value));
    }

    public InvocationNode inherit(final Node key) {
      return transform(key, key);
    }

    public InvocationNode inherit(final Serializable key) {
      return inherit(node(key));
    }

    public InvocationNode exceptionHandler(final Node exceptionHandler) {
      properties.put(node(Common.exceptionHandler), exceptionHandler);
      return this;
    }

    /**
     * Convenience method that automatically connects prerequisite invocations. This
     * connects the invocation nodes listed as transform dependencies directly
     * (rather than their EAV nodes), so it does not cover contextual parameters
     * (which should be predicated on the parent entrypoint instead) or local
     * variables.
     */
    public InvocationNode autoConnect() {
      conjunction(properties.get(node(Common.transform)).properties.values()
          .stream()
          .filter(node -> node instanceof InvocationNode)
          .toArray(Node[]::new));
      return this;
    }

    /**
     * Builder version of {@link Node#then(SynapticNode)}.
     */
    public InvocationNode next(final SynapticNode next) {
      then(next);
      return this;
    }

    public InvocationNode afterAll(final Node node) {
      properties.put(node(Common.afterAll), node);
      return this;
    }

    public InvocationNode forEach(final Node node) {
      properties.put(node(Common.forEach), node);
      return this;
    }
  }

  public static CompletableFuture<Void> reinforceRecursively(final Context context, final Optional<Long> decayPeriod,
      final float weight) {
    if (context instanceof KbContext kbContext) {
      return kbContext.reinforceRecursively(decayPeriod, weight);
    } else {
      return context.reinforce(decayPeriod, weight);
    }
  }

  public static CompletableFuture<Void> reinforceRecursively(final Context context, final float weight) {
    return reinforceRecursively(context, Optional.empty(), weight);
  }

  public static void shutDownRecursively(final Context context) {
    if (context instanceof KbContext kbContext) {
      kbContext.shutDownRecurisvely();
    } else {
      context.getScheduler()
          .shutdown();
    }
  }

  private static void writeIndex(final Map<Object, SynapticNode> index, final ObjectOutputStream stream)
      throws IOException {
    // Discard keys that aren't serializable.
    for (val entry : index.entrySet()) {
      if (entry.getKey() instanceof Serializable) {
        stream.writeObject(entry.getKey());
        stream.writeObject(entry.getValue());
      }
    }
    stream.writeObject(null);
  }

  private void writeObject(final ObjectOutputStream stream) throws IOException {
    stream.defaultWriteObject();
    writeIndex(strongIndex, stream);
    writeIndex(weakIndex, stream);
  }

  private static void readIndex(final Map<Object, SynapticNode> index, final ObjectInputStream stream)
      throws IOException, ClassNotFoundException {
    Object key;
    while ((key = stream.readObject()) != null) {
      index.put(key, (SynapticNode) stream.readObject());
    }
  }

  private void readObject(final ObjectInputStream stream) throws ClassNotFoundException, IOException {
    preInit();
    stream.defaultReadObject();

    readIndex(strongIndex, stream);
    readIndex(weakIndex, stream);

    val it = eavNodes.entrySet()
        .iterator();
    while (it.hasNext()) {
      val entry = it.next();

      // Tuples that were serialized with an unreachable element are marked by a null
      // array.
      if (entry.getKey().tuple == null) {
        it.remove();
      } else {
        entry.getKey()
            .makeCanonical();
      }
    }
    postInit();
  }

  @Override
  public void close() {
    eavCleanupThread.interrupt();
    rxOutput.onComplete();
    threadPool.shutdown();
  }

  public Observable<String> rxOutput() {
    return rxOutput;
  }

  /**
   * Immutable class labeling a positional data element. This class is marked
   * immutable by the {@code KnowledgeBase}.
   */
  public static record Ordinal(Node type, int ordinal) implements Serializable {
    private static final long serialVersionUID = -2554583140984988117L;

    @Override
    public String toString() {
      return String.format("%s[%s]", type, ordinal);
    }
  }

  /**
   * Gets or creates a node representing a positional argument.
   * 
   * @param ordinal 0-based argument index
   * @return "arg<i>n</i>"
   */
  public Node arg(final int ordinal) {
    return node(new Ordinal(node(Common.argument), ordinal));
  }

  /**
   * Gets or creates a node representing a positional parameter type, for use with
   * {@link BuiltIn#invoke}.
   * 
   * @param ordinal 0-based argument index
   * @return "param<i>n</i>"
   */
  public Node param(final int ordinal) {
    return node(new Ordinal(node(Common.parameter), ordinal));
  }

  private Map<EavTuple, Node> eavNodes = new ConcurrentHashMap<>();
  private transient ReferenceQueue<Node> eavCleanup;
  private transient Thread eavCleanupThread;

  private void preInitEav() {
    eavCleanup = new ReferenceQueue<>();
  }

  private void postInitEav() {
    eavCleanupThread = new Thread(() -> {
      try {
        while (!Thread.interrupted()) {
          ((EavTuple.CanonicalWeakReference) eavCleanup.remove()).destroyTuple();
        }
      } catch (InterruptedException e) {
      }
    });
    eavCleanupThread.start();
  }

  public Node eavNode(final boolean relative, final boolean wildcard, final Node... tuple) {
    return eavNodes.computeIfAbsent(new EavTuple(relative, wildcard, tuple), t -> {
      t.makeCanonical();
      return new Node();
    });
  }

  private void activateEavNodes(final Context context, Node object, final Node property, final Node value) {
    // EAV nodes are meant to capture specializations and generalizations, so EAV
    // nodes for ephemeral context values defeat the purpose and accumulate garbage
    // that would require nontrivial analysis to clean up (since they could
    // potentialy affect other state once they are connected through Hebbian
    // learning, it is not obvious in what cases it would be safe to cull them).
    // Thus, go ahead and omit those EAV nodes.
    if (object != null && object.getValue() instanceof Context || value != null && value.getValue() instanceof Context)
      return;
    // (The case of property.value instanceof context is not covered because what
    // does that even mean, and there would probably be an interesting reason for
    // doing something like that.)

    // In addition to top-level contextual nodes, we also activate EAV nodes for the
    // first layer of properties of a contextual node. This is because there's no
    // other way to watch "references" to nodes.

    // For now, we don't attempt to activate EAV nodes across contexts, even if
    // we're setting properties on a context node. This may change in the future.
    // For now it can be seen kind of like synchronization in Java.

    if (object == null) {
      // It's important to activate specific nodes first and then wildcard nodes so
      // that we can inhibit on particular values. By the same logic, activate
      // properties before main nodes.
      if (value != null) {
        // We need to collect prior to activating since we're iterating over properties
        // and activation could alter the context.
        val propEavs = new ArrayList<Node>(2 * value.properties.size());
        synchronized (value.properties) {
          for (val entry : value.properties.entrySet()) {
            propEavs.add(eavNode(true, false, property, entry.getKey(), entry.getValue()));
            propEavs.add(eavNode(true, true, property, entry.getKey()));
          }
        }
        for (val node : propEavs) {
          node.activate(context);
        }
      }

      eavNode(true, false, property, value).activate(context);
      eavNode(true, true, property).activate(context);

    } else {
      eavNode(false, false, object, property, value).activate(context);
      eavNode(false, true, object, property).activate(context);

      // Also find contextual references.
      // We need to collect prior to activating since we're iterating over properties
      // and activation could alter the context.
      val refEavs = new ArrayList<Node>();
      synchronized (context.node.properties) {
        for (val entry : context.node.properties.entrySet()) {
          if (entry.getValue() == object) {
            refEavs.add(eavNode(true, false, entry.getKey(), property, value));
            refEavs.add(eavNode(true, true, entry.getKey(), property));
          }
        }
      }
      for (val node : refEavs) {
        node.activate(context);
      }
    }
  }

  private final class EavTuple implements Serializable {
    private static final long serialVersionUID = 5006832330252764773L;

    class CanonicalWeakReference extends WeakReference<Node> {
      CanonicalWeakReference(final Node node) {
        super(node, eavCleanup);
      }

      void destroyTuple() {
        destroy();
      }
    }

    transient WeakReference<Node>[] tuple;
    transient int hashCode;
    final boolean relative, wildcard;

    EavTuple(final boolean relative, final boolean wildcard, final Node... tuple) {
      this.relative = relative;
      this.wildcard = wildcard;
      init(tuple);
    }

    @SuppressWarnings("unchecked")
    void init(final Node[] tuple) {
      this.tuple = new WeakReference[tuple.length];
      for (int i = 0; i < tuple.length; ++i) {
        this.tuple[i] = tuple[i] == null ? null : new WeakReference<>(tuple[i]);
      }
      hashCode = Arrays.deepHashCode(new Object[] { relative, wildcard, tuple });
    }

    void makeCanonical() {
      for (int i = 0; i < tuple.length; ++i) {
        if (tuple[i] != null) {
          tuple[i] = new CanonicalWeakReference(tuple[i].get());
        }
      }
    }

    private void writeObject(final ObjectOutputStream stream) throws IOException {
      stream.defaultWriteObject();

      final Node[] tuple = new Node[this.tuple.length];
      for (int i = 0; i < tuple.length; ++i) {
        val ref = this.tuple[i];
        if (ref == null) {
          tuple[i] = null;
        } else {
          val node = ref.get();
          if (node == null) {
            // This will be evicted after deserialization.
            stream.writeObject(null);
            return;
          } else {
            tuple[i] = node;
          }
        }
      }
      stream.writeObject(tuple);
    }

    private void readObject(final ObjectInputStream stream) throws ClassNotFoundException, IOException {
      stream.defaultReadObject();

      val tuple = (Node[]) stream.readObject();
      if (tuple != null)
        init(tuple);
    }

    void destroy() {
      // Note that since this happens in response to a tuple element having become
      // unreachable, we don't really have to worry much about race conditions since
      // no write conflicts can arise at this point.
      eavNodes.remove(this);
      for (val ref : tuple) {
        if (ref != null)
          ref.clear();
      }
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this)
        return true;

      if (!(obj instanceof EavTuple other))
        return false;

      // eviction across serialization boundaries
      if (tuple == null || other.tuple == null)
        return false;

      if (relative != other.relative || wildcard != other.wildcard)
        return false;

      if (tuple.length != other.tuple.length)
        return false;

      for (int i = 0; i < tuple.length; ++i) {
        if (tuple[i] == other.tuple[i])
          continue;
        if (tuple[i] == null || other.tuple[i] == null || tuple[i].get() != other.tuple[i].get())
          return false;
      }

      return true;
    }

    @Override
    public int hashCode() {
      return hashCode;
    }
  }

  /**
   * @param context the context to be used if {@code object} is null. This is also
   *                the context in which EAV nodes are activated.
   * @return the previous value of the property, if any.
   */
  private Node setProperty(final Context context, final Node object, final Node property, final Node value) {
    final Node old;

    if (object == null) {
      if (value == null) {
        old = context.node.properties.remove(property);
      } else {
        old = context.node.properties.put(property, value);
      }

      if (property.getValue() == Common.returnValue) {
        final Node caller = optionalNode(context, Common.caller);
        if (caller != null) {
          final Node invocation = caller.properties.get(node(Common.invocation));
          if (invocation != null) {
            val parent = (Context) caller.properties.get(node(Common.context))
                .getValue();
            setProperty(parent, null, invocation, value);

            // Null list elements can't be packed/unpacked naively (an absent ordinal is a
            // sentinel) and there are other reasons for excluding them from collection
            // interpretations of return values.
            if (value != null) {
              val afterAll = invocation.properties.get(node(Common.afterAll));
              if (afterAll != null) {
                parent.addReturnValue(afterAll, value);
              }

              val forEach = invocation.properties.get(node(Common.forEach));
              if (forEach != null) {
                val childContext = newContext();
                setProperty(childContext, null, node(Common.caller), optionalNode(parent, Common.caller));
                setProperty(childContext, null, node(Common.maxStackDepth), optionalNode(parent, Common.maxStackDepth));
                // This could conceivably be the invocation node as well for isomorphicity with
                // normal return.
                setProperty(childContext, null, node(Common.value), value);
                inheritActivity(parent, childContext);
                childContext.exceptionHandler = parent.exceptionHandler;
                forEach.activate(childContext);
              }
            }
          }
        }
        context.continuation()
            .complete(null);
      }
    } else {
      if (value == null) {
        old = object.properties.remove(property);
      } else {
        old = object.properties.put(property, value);
      }
    }

    activateEavNodes(context, object, property, value);

    return old;
  }

  // A type is marked immutable by its class node having a value property with a
  // canonicalizing map.
  private void markImmutable(Class<?> type) {
    node(type).properties.put(node(Common.value), node(new ConcurrentHashMap<Serializable, Serializable>()));
  }

  // The index doesn't allow nulls and we can short-circuit most of our logic, so
  // special-case this here.
  private final SynapticNode nullNode = new SynapticNode();

  /**
   * Single-instance resolution for immutable types. Skip for/bootstrap with Class
   * and enum since they're always single-instance.
   * 
   * An immutable class is designated by having the node for the class instance
   * have a Common.value property that has a value of a ConcurrentHashMap, which
   * will be used to resolve values to canonical instances.
   */
  @SuppressWarnings("unchecked")
  private Serializable canonicalize(Serializable value) {
    if (value instanceof Class<?> || value instanceof Enum)
      return value;

    final Node values = node(value.getClass()).properties.get(node(Common.value));
    if (values != null && values.getValue() instanceof ConcurrentHashMap) {
      return ((ConcurrentHashMap<Serializable, Serializable>) values.getValue()).computeIfAbsent(value, x -> value);
    } else {
      return value;
    }
  }

  public SynapticNode node(Object value) {
    if (value == null)
      return nullNode;

    if (value instanceof Serializable s) {
      value = canonicalize(s);
    }

    final Map<Object, SynapticNode> index = value instanceof Context ? weakIndex : strongIndex;
    return index.computeIfAbsent(value, v -> v instanceof BuiltIn b ? b.node(this) : new SynapticNode(v));
  }

  /**
   * A predicate for {@link Node#debugDump(Predicate)} that skips caller records,
   * which can bloat rapidly.
   */
  public static final Predicate<Map.Entry<Node, Node>> SKIP_CALLER = e -> e.getKey()
      .getValue() != Common.caller;

  /**
   * Adds a debug listener that dumps the context when a node is activated.
   */
  public static void debug(final Node node) {
    node.rxActivate()
        .subscribe(a -> System.err.printf("Watch %s context: %s\n", node, a.context.node.debugDump(SKIP_CALLER)));
  }

  public void indexNode(final Node index, final Node node, final String name) {
    node.comment = name;
    index.properties.put(node(name), node);
  }

  // The ordering of the members is significant; later setups can depend on
  // earlier ones.
  public enum Bootstrap {
    /**
     * <li>{@link Common#javaClass}: class for which to invoke default constructor.
     * <li>Returns the instance that was created.
     */
    newInstance {
      @Override
      protected void setUp(final KnowledgeBase kb, final Node newInstance) {
        val call = kb.new InvocationNode(BuiltIn.method)
            .literal(Common.name, "newInstance")
            .transform(Common.object, Common.javaClass)
            .conjunction(newInstance);

        kb.new InvocationNode(BuiltIn.setProperty)
            .literal(Common.name, Common.returnValue)
            .transform(Common.value, call)
            .autoConnect();
      }
    },
    /**
     * <li>{@link Common#javaClass}: class to mark immutable
     */
    markImmutable {
      @Override
      protected void setUp(final KnowledgeBase kb, final Node markImmutable) {
        val newMap = kb.new InvocationNode(newInstance)
            .literal(Common.javaClass, ConcurrentHashMap.class)
            .conjunction(markImmutable);

        kb.new InvocationNode(BuiltIn.setProperty)
            .literal(Common.name, Common.value)
            .transform(Common.object, Common.javaClass)
            .transform(Common.value, newMap)
            .autoConnect();

        // Mark some more common types immutable.
        for (final Class<?> type : new Class<?>[] { Boolean.class, Byte.class, Character.class, Integer.class,
            Long.class, Float.class, Double.class }) {
          val context = kb.newContext();
          kb.new InvocationNode(markImmutable)
              .literal(Common.javaClass, type)
              .activate(context);
          context.blockUntilIdle();
        }
      }
    },
    getOrInherit {
      @Override
      protected void setUp(final KnowledgeBase kb, final Node getOrInherit) {
        val getParentContext = kb.new InvocationNode(BuiltIn.getProperty)
            .transform(Common.object, Common.caller)
            .literal(Common.name, Common.context)
            .conjunction(getOrInherit);

        val getInContext = kb.new InvocationNode(BuiltIn.getProperty)
            .transform(Common.object, getParentContext)
            .inherit(Common.name)
            .autoConnect();

        val absentInContext = kb.eavNode(true, false, getInContext, null);

        kb.new InvocationNode(BuiltIn.setProperty)
            .literal(Common.name, Common.returnValue)
            .transform(Common.value, getInContext)
            .autoConnect()
            .inhibitor(absentInContext);

        val getGrandcaller = kb.new InvocationNode(BuiltIn.getProperty)
            .transform(Common.object, getParentContext)
            .literal(Common.name, Common.caller)
            .conjunction(absentInContext);

        val getGrandcontext = kb.new InvocationNode(BuiltIn.getProperty)
            .transform(Common.object, getGrandcaller)
            .literal(Common.name, Common.context)
            .autoConnect();

        val getInherited = kb.new InvocationNode(BuiltIn.getProperty)
            .transform(Common.object, getGrandcontext)
            .inherit(Common.name)
            .autoConnect();

        kb.new InvocationNode(BuiltIn.setProperty)
            .literal(Common.name, Common.returnValue)
            .transform(Common.value, getInherited)
            .autoConnect();
      }
    },
    /**
     * Index of built-in nodes by string name.
     */
    index {
      private <T extends Enum<T>> SynapticNode compileEnum(final KnowledgeBase kb, final T[] e) {
        val index = new SynapticNode();
        for (final T member : e) {
          index.properties.put(kb.node(member.name()), kb.node(member));
        }
        return index;
      }

      @Override
      protected void setUp(final KnowledgeBase kb, final Node index) {
        val common = compileEnum(kb, Common.values()), builtIn = compileEnum(kb, BuiltIn.values()),
            bootstrap = compileEnum(kb, Bootstrap.values());
        for (val type : new SynapticNode[] { common, builtIn, bootstrap }) {
          index.properties.putAll(type.properties);
        }
        index.properties.put(kb.node("index"), index);
        index.properties.put(kb.node("Common"), common);
        index.properties.put(kb.node("BuiltIn"), builtIn);
        index.properties.put(kb.node("Bootstrap"), bootstrap);
      }
    },
    /**
     * Rewrite entrypoint called by {@link #incrementalParse}. When this is called
     * {@link Common#rewriteWindow} is in the context and its n-gram EAV nodes have
     * been dispatched. This should return new rewrite windows.
     */
    processRewrite,
    /**
     * Calls {@link BuiltIn#dispatchNGrams} on {@link Common#rewriteWindow} and
     * recurses on the resulting return values from {@link #processRewrite}, which
     * should be new rewrite windows. When the end of a rewrite window is reached,
     * returns the single element or errors.
     */
    incrementalParse {
      @Override
      protected void setUp(final KnowledgeBase kb, final Node incrementalParse) {
        val recurse = kb.new InvocationNode(incrementalParse)
            .transform(Common.rewriteWindow, Common.value)
            .inherit(Common.caller);
        val dispatch = kb.new InvocationNode(kb.new InvocationNode(BuiltIn.dispatchNGrams)
            .inherit(Common.rewriteWindow)
            .next(kb.node(Bootstrap.processRewrite)))
                .inherit(Common.rewriteWindow)
                .forEach(recurse);
        incrementalParse.then(dispatch);

        // Additionally, we need to advance the window if no rewrites were attempted.
        val afterDispatch = new SynapticNode();
        dispatch.afterAll(afterDispatch);
        val advance = kb.new InvocationNode(BuiltIn.rewrite)
            .inherit(Common.rewriteWindow)
            .literal(Common.rewriteLength, 0);
        afterDispatch.then(advance)
            .inhibitor(kb.eavNode(true, true, kb.node(new Ordinal(afterDispatch, 0))));
        advance.forEach(recurse);

        val getResult = kb.new InvocationNode(BuiltIn.method)
            .transform(Common.object, Common.rewriteWindow)
            .literal(Common.name, "single");
        kb.eavNode(true, false, advance, null)
            .then(getResult);
        kb.new InvocationNode(BuiltIn.setProperty)
            .literal(Common.name, Common.returnValue)
            .transform(Common.value, getResult)
            .autoConnect();
      }
    },
    /**
     * Interpretation is split into parse and eval phases. As language becomes more
     * natural they blur, but a rough demarcation is that parse should not have
     * lasting side effects (perhaps aside from altering a parse context).
     * 
     * <ul>
     * <li>{@link Common#value}: string to parse
     * </ul>
     */
    parse {
      @Override
      protected void setUp(final KnowledgeBase kb, final Node parse) {
        val rewriter = kb.new InvocationNode(BuiltIn.createRewriteWindow).inherit(Common.value);
        parse.then(rewriter);
        kb.new InvocationNode(Bootstrap.incrementalParse)
            .transform(Common.rewriteWindow, rewriter)
            .autoConnect()
            .forEach(kb.new InvocationNode(BuiltIn.setProperty)
                .literal(Common.name, Common.returnValue)
                .inherit(Common.value));
      }
    },
    /**
     * <li>{@link Common#value}: string to evaluate
     */
    eval {
      @Override
      protected void setUp(final KnowledgeBase kb, final Node eval) {
        val parse = kb.new InvocationNode(Bootstrap.parse).inherit(Common.value);
        eval.then(parse);

        val parentContext = kb.new InvocationNode(BuiltIn.getProperty)
            .transform(Common.object, Common.caller)
            .literal(Common.name, Common.context);
        parse.then(parentContext);

        kb.new InvocationNode(BuiltIn.activate)
            .transform(Common.value, parse)
            .transform(Common.context, parentContext)
            .autoConnect();
      }
    };

    protected void setUp(final KnowledgeBase kb, final Node node) {
    }

    @Override
    public String toString() {
      return toQualifiedString(this);
    }
  }

  // Creates common support routines and marks basic immutable types.
  private void bootstrap() {
    // Since strings are used for reflection at all and markImmutable is
    // implemented in terms of reflection, we need to bootstrap strings as
    // immutable.
    markImmutable(String.class);
    // We also need to mark args and params as immutable as they're potentially
    // touched in the markImmutable invocation.
    markImmutable(Ordinal.class);

    for (val bootstrap : Bootstrap.values()) {
      bootstrap.setUp(this, node(bootstrap));
    }
  }
}
