package ai.xng;

import static ai.xng.KnowledgeBase.POP_FACTOR;
import static ai.xng.KnowledgeBase.PUSH_FACTOR;

import java.util.Iterator;
import java.util.Optional;

import com.google.common.collect.ImmutableList;

import ai.xng.constructs.BooleanDecoder;
import ai.xng.constructs.CharacterDecoder;
import ai.xng.constructs.CoincidentEffect;
import ai.xng.constructs.Latch;
import lombok.AllArgsConstructor;
import lombok.val;

public class LanguageBootstrap {
  private final KnowledgeBase kb;

  private <H extends Prior> Sequence<H> asSequence(final H start) {
    return new Sequence<>(start, start);
  }

  @AllArgsConstructor
  private class Sequence<H extends Node> {
    H head;
    Prior tail;

    public Sequence<H> thenDirect(final BiNode next, final IntegrationProfile profile) {
      tail.then(next, profile);
      return new Sequence<>(head, next);
    }

    public Sequence<H> then(final Posterior... p) {
      tail = tail.then(kb.execution.new Node());
      for (val e : p) {
        tail.then(e);
      }
      return this;
    }

    public Sequence<H> thenDelay() {
      tail = tail.then(kb.execution.new Node());
      return this;
    }

    public Sequence<H> stanza() {
      return then(control.resetStanza)
          .thenDelay()
          .thenDelay();
    }
  }

  // TODO: For compatibility with capture, spawn should copy the integrator state
  // from the spawn node to the node spawned.
  private class Spawn {
    final ActionCluster.Node stateRecognition = kb.actions.new Node(() -> kb.stateRecognition.new Node().activate()),
        context = kb.actions.new Node(() -> kb.context.new Node().activate()),
        sequenceRecognition = kb.actions.new Node(() -> kb.sequenceRecognition.new Node().activate()),
        data = kb.actions.new Node(() -> kb.data.new MutableNode<>().activate());
  }

  private final Spawn spawn;

  private class Control {
    final ImmutableList<Cluster.PriorClusterProfile> frameFieldPriors = new Cluster.PriorClusterProfile.ListBuilder()
        .add(kb.context)
        .add(kb.naming)
        .build();

    final StmCluster stackFrame = new StmCluster("stackFrame"),
        returnValue = new StmCluster("returnValue"),
        cxt = new StmCluster("cxt"),
        tmp = new StmCluster("tmp");
    final BiCluster.Node staticContext = kb.naming.new Node("staticContext"),
        entrypoint = kb.naming.new Node("entrypoint"),
        arg1 = kb.naming.new Node("arg1"),
        returnTo = kb.naming.new Node("returnTo");
    final BiCluster.Node execute = kb.entrypoint.new Node("execute"),
        doReturn = kb.entrypoint.new Node("doReturn");

    final BiCluster.Node resetStanza = kb.execution.new Node();

    void setUp() {
      resetStanza.then(
          kb.resetOutboundPosteriors(kb.execution),
          kb.resetPosteriors(stackFrame),
          kb.resetPosteriors(returnValue),
          kb.resetPosteriors(cxt),
          kb.resetPosteriors(tmp),
          kb.resetPosteriors(kb.naming),
          kb.resetPosteriors(kb.context),
          kb.resetPosteriors(kb.entrypoint));

      asSequence(execute)
          .stanza()
          .then(stackFrame.address)
          .then(staticContext)
          .then(entrypoint);

      asSequence(doReturn)
          .stanza()
          .then(stackFrame.address)
          .then(kb.scalePosteriors(stackFrame, POP_FACTOR), returnTo)
          .then(kb.disassociate(stackFrame, kb.context));
    }
  }

  private final Control control;

  private class StringIterator {
    final BiNode create = kb.execution.new Node();
    /**
     * Node activated once a code point has been decoded.
     */
    final BiNode onNext = kb.execution.new Node();
    /**
     * Node that should be called once a longer processing operation is ready to
     * advance the iterator. It can also be inhibited by paths that are not ready to
     * proceed.
     */
    final BiNode advance = kb.entrypoint.new Node();
    final BooleanDecoder hasNextDecoder = new BooleanDecoder(kb.actions, kb.data, kb.input,
        data -> data instanceof Iterator<?> i ? Optional.of(i.hasNext()) : Optional.empty());
    final BiNode codePoint = kb.naming.new Node();
    final InputCluster charCluster = new InputCluster();
    final CharacterDecoder charDecoder = new CharacterDecoder(kb.actions, kb.data, charCluster);

    void setUp() {
      val iterator = kb.naming.new Node();

      val iterator_in = new CoincidentEffect.Curry<>(kb.actions, kb.data),
          iterator_out = new CoincidentEffect.Curry<>(kb.actions, kb.data);

      asSequence(create)
          .stanza()
          .then(control.stackFrame.address)
          .then(iterator)
          // TODO: To use capture with spawn, we need to activate the spawned node's
          // integrator.
          .then(spawn.data)
          .then(kb.associate(control.frameFieldPriors, kb.data))

          .stanza()
          .then(control.stackFrame.address)
          .then(control.arg1)
          .then(iterator_in.node)

          .stanza()
          .then(control.stackFrame.address)
          .then(iterator)
          .then(iterator_out.node)
          .then(kb.actions.new Node(() -> ((DataCluster.MutableNode<Object>) iterator_out.require()).setData(
              (((String) iterator_in.require().getData()).codePoints().iterator()))))

          .stanza()
          .then(control.stackFrame.address)
          .then(codePoint)
          .then(spawn.data)
          .then(kb.associate(control.frameFieldPriors, kb.data))

          .then(advance);

      asSequence(advance)
          .then(kb.resetPriors(kb.entrypoint))
          .stanza()
          .then(control.stackFrame.address)
          .then(iterator)
          .then(hasNextDecoder.node);

      val next_in = new CoincidentEffect.Curry<>(kb.actions, kb.data),
          next_out = new CoincidentEffect.Curry<>(kb.actions, kb.data);

      asSequence(hasNextDecoder.isTrue)
          .stanza()
          .then(control.stackFrame.address)
          .then(iterator)
          .then(next_in.node)

          .stanza()
          .then(control.stackFrame.address)
          .then(codePoint)
          .then(next_out.node)
          .then(kb.actions.new Node(() -> ((DataCluster.MutableNode<Integer>) next_out.require()).setData(
              ((Iterator<Integer>) next_in.require().getData()).next())))

          .stanza()
          .then(control.stackFrame.address)
          .then(codePoint)
          .then(charDecoder.node)
          .then(onNext)
          // TODO: We might consider binding codePoint to a register first to avoid
          // polluted state during recognition.
          // Advance by default unless inhibited.
          .thenDirect(advance, IntegrationProfile.PERSISTENT);
    }
  }

  private final StringIterator stringIterator;

  private class RecognitionClass {
    final BiCluster.Node character = kb.stateRecognition.new Node();

    void setUp() {
      // character recognition capture
      // We expect recognized characters to trigger a recognition tag two nodes deep,
      // with the first being the capture itself.
      val captureDispatch = stringIterator.onNext
          .then(kb.execution.new Node())
          .then(kb.execution.new Node()) // recognition would trigger here
          .then(kb.execution.new Node());
      captureDispatch.inhibitor(character);
      captureDispatch.then(kb.actions.new Node(() -> {
        val capture = kb.stateRecognition.new Node();
        capture.then(character);
        capture.activate();
      }));
      captureDispatch
          .then(kb.execution.new Node())
          .then(kb.associate(stringIterator.charCluster, kb.stateRecognition));
    }
  }

  private final RecognitionClass recognitionClass;

  /**
   * This class modifies the InputIterator to capture a recognition conjunction
   * for every frame while active. It does not itself form an association from the
   * captured recognition.
   * <p>
   * Typical usage of this utility is to immediately bind the captured recognition
   * to a recognition circuit, which includes a familiarity tag, semantics, and
   * binding.
   */
  private class RecognitionSequenceMemorizer {
    final BiNode staticContext = kb.context.new Node("rsm"),
        entrypoint = kb.entrypoint.new Node("rsm/entrypoint");

    void setUp() {
      entrypoint.conjunction(staticContext, control.entrypoint);
      asSequence(entrypoint)
          // TODO: It may be more idiomatic to create the capture later, but with current
          // limitations that would not allow us to bind to STM while also binding to the
          // captured sequence.
          .then(control.returnValue.address, kb.suppressPosteriors(control.returnValue))
          .then(spawn.stateRecognition, kb.clearPosteriors(control.returnValue))
          .then(kb.associate(control.returnValue, kb.stateRecognition))
          .then(stringIterator.create);

      // This is shared with String LiteralBuilder
      recognitionClass.character.then(control.stackFrame.address, control.staticContext);

      // Hook sequence capture up after character capture to avoid dealing with the
      // input conjunction directly.
      val capture = kb.execution.new Node();
      capture.conjunction(recognitionClass.character, staticContext);
      asSequence(capture)
          .then(spawn.sequenceRecognition)
          .then(kb.associate()
              .priors(kb.sequenceRecognition, IntegrationProfile.TWOGRAM)
              .priors(kb.stateRecognition)
              .to(kb.sequenceRecognition));

      val captureReturn = kb.execution.new Node();
      stringIterator.hasNextDecoder.isFalse.then(control.staticContext);
      captureReturn.conjunction(stringIterator.hasNextDecoder.isFalse, staticContext);
      asSequence(captureReturn)
          .stanza()
          .then(control.returnValue.address)
          .then(kb.capture()
              .baseProfiles(IntegrationProfile.TWOGRAM)
              .priors(kb.sequenceRecognition)
              .to(kb.stateRecognition))
          .then(control.doReturn);
    }
  }

  public final RecognitionSequenceMemorizer recognitionSequenceMemorizer;

  private class Parse {
    final BiNode staticContext = kb.context.new Node("parse"),
        entrypoint = kb.entrypoint.new Node("parse/entrypoint");

    final BiCluster.Node constructionPointer = kb.naming.new Node("constructionPointer"),
        writePointer = kb.naming.new Node("writePointer");

    void setUp() {
      entrypoint.conjunction(staticContext, control.entrypoint);
      // One of the first things we should do when we begin parsing something is start
      // constructing a stack frame.
      asSequence(entrypoint)
          .stanza()
          .then(control.stackFrame.address)
          .then(constructionPointer)
          .then(spawn.context)
          .then(kb.associate(control.frameFieldPriors, kb.context))
          .then(stringIterator.create);

      val print = kb.context.new Node("print"),
          printEntrypoint = kb.entrypoint.new Node("print/entrypoint");
      printEntrypoint.conjunction(print, control.entrypoint);
      asSequence(printEntrypoint)
          .stanza()
          .then(control.stackFrame.address)
          .then(control.arg1)
          .then(kb.print)
          .then(control.doReturn);

      val bindPrintEntrypoint = kb.entrypoint.new Node();
      bindPrintEntrypoint.then(control.stackFrame.address, control.staticContext);

      val bindPrint = kb.execution.new Node();
      bindPrint.conjunction(bindPrintEntrypoint, staticContext);
      bindPrint.inhibit(stringIterator.advance, IntegrationProfile.PERSISTENT);
      asSequence(bindPrint)
          .stanza()
          .then(control.stackFrame.address)
          .then(constructionPointer, control.cxt.address, kb.suppressPosteriors(control.cxt))
          .then(kb.capture(control.cxt, kb.context))
          .then(kb.actions.new Node(() -> System.out.println(control.cxt.address.getPosteriors())))

          .stanza()
          .then(control.cxt.address)
          .then(control.staticContext)
          .then(print)
          .then(kb.associate(control.frameFieldPriors, kb.context))

          // In the future we might like to scope the write pointer per construction
          // frame, but that story is not fleshed out yet so let's keep it simple for now.
          .stanza()
          .then(control.stackFrame.address)
          .then(writePointer)
          .then(control.arg1)
          .then(kb.associate(control.frameFieldPriors, kb.naming))
          .then(stringIterator.advance);

      val returnParseFrame = kb.execution.new Node();
      // shared with recognition sequence memorizer
      stringIterator.hasNextDecoder.isFalse.then(control.staticContext);
      returnParseFrame.conjunction(stringIterator.hasNextDecoder.isFalse, staticContext);
      asSequence(returnParseFrame)
          .stanza()
          .then(control.stackFrame.address)
          .then(constructionPointer, control.returnValue.address, kb.suppressPosteriors(control.returnValue))
          .then(kb.clearPosteriors(control.returnValue))
          .then(kb.associate(control.returnValue, kb.context))
          .then(control.doReturn);

      {
        val call = kb.context.new Node("train: \"print(\"");
        recognitionSequenceMemorizer.staticContext.conjunction(call, control.staticContext);
        kb.data.new FinalNode<>("print(").conjunction(call, control.arg1);
        val bindBindPrint = kb.execution.new Node();
        bindBindPrint.conjunction(call, control.returnTo);
        asSequence(bindBindPrint)
            .then(control.returnValue.address)
            .thenDelay()
            .then(bindPrintEntrypoint)
            .then(kb.associate(kb.stateRecognition, kb.entrypoint));
        control.stackFrame.address.then(call);
        control.execute.activate();
        Scheduler.global.fastForwardUntilIdle();
        Scheduler.global.fastForwardFor(IntegrationProfile.PERSISTENT.period());
      }
    }
  }

  private final Parse parse;

  // Eval parses the argument and executes the resulting construct.
  private class Eval {
    final BiNode staticContext = kb.context.new Node("eval"),
        entrypoint = kb.entrypoint.new Node("eval/entrypoint");

    void setUp() {
      entrypoint.conjunction(staticContext, control.entrypoint);

      val executeParsed = kb.entrypoint.new Node("eval/executeParsed");
      asSequence(executeParsed)
          .stanza()
          .then(control.stackFrame.address, control.returnValue.address, kb.suppressPosteriors(control.stackFrame))
          .then(kb.scalePosteriors(control.stackFrame, PUSH_FACTOR))
          .then(kb.associate(control.stackFrame, kb.context))

          .stanza()
          .then(control.stackFrame.address)
          .then(control.returnTo)
          // Note that at some point, we have considered expanding suppressPosteriors to
          // also suppress the trace, for use in pulling down nodes after activation to
          // clear working memory. If we do end up needing that behavior, we will need to
          // reconcile against this use case where we suppress posteriors in order to
          // highlight a node's trace without side effects.
          .then(control.doReturn, kb.suppressPosteriors(kb.entrypoint))
          .then(kb.associate(control.frameFieldPriors, kb.entrypoint))
          .then(control.execute);

      asSequence(entrypoint)
          .stanza()
          .then(control.cxt.address, kb.suppressPosteriors(control.cxt))
          .then(spawn.context, kb.clearPosteriors(control.cxt))
          .then(kb.associate(control.cxt, kb.context))

          .stanza()
          .then(control.cxt.address)
          .then(control.staticContext)
          .then(parse.staticContext)
          .then(kb.associate(control.frameFieldPriors, kb.context))

          .stanza()
          .then(control.tmp.address)
          .then(kb.clearPosteriors(control.tmp))

          .stanza()
          .then(control.stackFrame.address)
          .then(control.arg1, control.tmp.address)
          .thenDelay()
          .then(kb.associate(control.tmp, kb.data))

          .stanza()
          .then(control.cxt.address)
          .then(control.arg1, control.tmp.address)
          .thenDelay()
          .then(kb.associate(control.frameFieldPriors, kb.data))

          .stanza()
          .then(control.cxt.address)
          .then(control.returnTo)
          .then(executeParsed, kb.suppressPosteriors(kb.entrypoint))
          .then(kb.associate(control.frameFieldPriors, kb.entrypoint))

          .stanza()
          .then(control.stackFrame.address, control.cxt.address, kb.suppressPosteriors(control.stackFrame))
          .then(kb.scalePosteriors(control.stackFrame, PUSH_FACTOR))
          .then(kb.associate(control.stackFrame, kb.context))
          .then(control.execute);
    }
  }

  private final Eval eval;

  private class StringLiteralBuilder {
    final Latch isParsing = new Latch(kb.actions, kb.input);

    void setUp() {
      val builder = new StringBuilder();

      val start = kb.execution.new Node(),
          append = kb.execution.new Node(),
          end = kb.execution.new Node();
      start.then(isParsing.set);
      end.then(isParsing.clear);

      val quote = kb.stateRecognition.new Node();
      val quoteDelayed = quote
          .then(kb.execution.new Node())
          .then(kb.execution.new Node());
      val conjunction = new ConjunctionJunction();
      stringIterator.charDecoder.forOutput('"', conjunction::add);
      conjunction.build(quote).then(recognitionClass.character);
      start.conjunction(quoteDelayed, isParsing.isFalse);
      end.conjunction(quoteDelayed, isParsing.isTrue);
      // Shared with RSM
      recognitionClass.character.then(control.stackFrame.address, control.staticContext);

      kb.actions.new Node(isParsing).conjunction(recognitionClass.character, parse.staticContext);

      start.then(kb.actions.new Node(() -> builder.setLength(0)));
      end.inhibit(stringIterator.advance, IntegrationProfile.PERSISTENT);
      asSequence(end)
          .stanza()
          .then(control.stackFrame.address)
          .then(parse.constructionPointer, control.cxt.address, kb.suppressPosteriors(control.cxt))
          .then(kb.clearPosteriors(control.cxt))
          .then(kb.associate(control.cxt, kb.context))

          .stanza()
          .then(control.stackFrame.address)
          .then(control.tmp.address, parse.writePointer, kb.suppressPosteriors(control.tmp))
          .then(kb.clearPosteriors(control.tmp))
          .then(kb.associate(control.tmp, kb.naming))

          .stanza()
          .then(control.tmp.address, control.cxt.address)
          .thenDelay()
          .then(kb.actions.new Node(() -> kb.data.new FinalNode<>(builder.toString()).activate()))
          .then(kb.associate(control.frameFieldPriors, kb.data))
          .then(stringIterator.advance);

      val notQuote = kb.stateRecognition.new Node();
      recognitionClass.character.then(notQuote).inhibitor(quote);
      append.conjunction(notQuote, isParsing.isTrue);
      append.inhibit(stringIterator.advance, IntegrationProfile.PERSISTENT);
      asSequence(append)
          .stanza()
          .then(control.stackFrame.address)
          .then(stringIterator.codePoint)
          .then(new CoincidentEffect.Lambda<>(kb.actions, kb.data, node -> {
            if (node.getData()instanceof Integer codePoint) {
              builder.appendCodePoint(codePoint);
            }
          }).node)
          .then(stringIterator.advance);
    }
  }

  private final StringLiteralBuilder stringLiteralBuilder;

  public LanguageBootstrap(final KnowledgeBase kb) {
    this.kb = kb;

    spawn = new Spawn();
    control = new Control();
    stringIterator = new StringIterator();
    recognitionClass = new RecognitionClass();
    recognitionSequenceMemorizer = new RecognitionSequenceMemorizer();
    parse = new Parse();
    eval = new Eval();
    stringLiteralBuilder = new StringLiteralBuilder();

    control.setUp();
    stringIterator.setUp();
    recognitionClass.setUp();
    recognitionSequenceMemorizer.setUp();
    parse.setUp();
    eval.setUp();
    stringLiteralBuilder.setUp();

    // When the input changes, we need to construct an eval call.
    asSequence(kb.inputValue.onUpdate)
        .then(control.stackFrame.address, kb.suppressPosteriors(control.stackFrame))
        .then(spawn.context, kb.scalePosteriors(control.stackFrame, PUSH_FACTOR))
        .then(kb.associate(control.stackFrame, kb.context))

        .stanza()
        .then(control.stackFrame.address)
        .then(control.staticContext)
        .then(eval.staticContext)
        .then(kb.associate(control.frameFieldPriors, kb.context))

        .stanza()
        .then(control.stackFrame.address)
        .then(control.arg1)
        .then(kb.inputValue)
        .then(kb.associate(control.frameFieldPriors, kb.data))
        .then(control.execute);
  }
}
