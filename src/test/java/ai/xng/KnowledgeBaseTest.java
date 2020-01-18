package ai.xng;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import ai.xng.KnowledgeBase.BuiltIn;
import ai.xng.KnowledgeBase.Common;
import lombok.val;

public class KnowledgeBaseTest {
  @Test
  public void testNodeCreation() {
    try (val kb = new KnowledgeBase()) {
      val a = kb.node(), b = kb.node();
      assertNotSame(a, b);
    }
  }

  @Test
  public void testEmptySerialization() throws Exception {
    assertNotNull(TestUtil.serialize(new KnowledgeBase()));
  }

  private static void testPrint(final KnowledgeBase kb) throws Exception {
    val monitor = new EmissionMonitor<>(kb.rxOutput());

    val context = new Context(kb::node);
    context.node.properties.put(kb.node(Common.value), kb.node("foo"));
    kb.node(BuiltIn.print).activate(context);
    assertEquals("foo", monitor.emissions().blockingFirst());
    // ensure that the context closes
    context.continuation().get(1, TimeUnit.SECONDS);
  }

  @Test
  public void testPrint() throws Exception {
    testPrint(new KnowledgeBase());
  }

  @Test
  public void testPrintAfterSerialization() throws Exception {
    testPrint(TestUtil.serialize(new KnowledgeBase()));
  }

  @Test
  public void testInvocationUtility() {
    try (val kb = new KnowledgeBase()) {
      val invocation = kb.new Invocation(kb.node(), kb.node(BuiltIn.print));
      assertEquals(1, invocation.node.properties.size());
      invocation.literal(kb.node(Common.value), kb.node("foo"));
      assertEquals(2, invocation.node.properties.size());
      assertEquals(1, invocation.node.properties.get(kb.node(Common.literal)).properties.size());
    }
  }

  private static void setUpPropGet(final KnowledgeBase kb) {
    final Node roses = kb.node("roses"), color = kb.node("color");
    roses.properties.put(color, kb.node("red"));

    final Node rosesAre = kb.node("roses are");
    kb.new Invocation(rosesAre, kb.node(BuiltIn.getProperty)).literal(kb.node(Common.object), roses)
        .literal(kb.node(Common.name), color);

    rosesAre.then(kb.new Invocation(kb.node(), kb.node(BuiltIn.print)).transform(kb.node(Common.value), rosesAre).node);

  }

  private static void assertPropGet(final KnowledgeBase kb) {
    val sanity1 = new EmissionMonitor<>(kb.node("roses are").rxActivate()),
        sanity2 = new EmissionMonitor<>(kb.node(BuiltIn.getProperty).rxActivate()),
        sanity3 = new EmissionMonitor<>(kb.node(BuiltIn.print).rxActivate());
    val valueMonitor = new EmissionMonitor<>(kb.rxOutput());

    final Context context = new Context(kb::node);
    kb.node("roses are").activate(context);
    assertTrue(sanity1.didEmit());
    assertTrue(sanity2.didEmit());
    assertTrue(sanity3.didEmit());
    assertEquals("red", valueMonitor.emissions().blockingFirst());
  }

  @Test
  public void testPrintProp() {
    try (val kb = new KnowledgeBase()) {
      setUpPropGet(kb);
      assertPropGet(kb);
    }
  }

  @Test
  public void testPrintReliability() throws InterruptedException {
    try (val kb = new KnowledgeBase()) {
      setUpPropGet(kb);
      for (int i = 0; i < 100; i++) {
        Thread.sleep(2 * Synapse.DEBOUNCE_PERIOD);
        assertPropGet(kb);
      }
    }
  }

  @Test
  public void testPrintPropAfterSerialization() throws Exception {
    try (val kb = new KnowledgeBase()) {
      setUpPropGet(kb);
      assertPropGet(TestUtil.serialize(kb));
    }
  }

  @Test
  public void testException() {
    try (final KnowledgeBase kb = new KnowledgeBase()) {
      final Node exceptionHandler = kb.node();
      val monitor = new EmissionMonitor<>(exceptionHandler.rxActivate());

      final Node invocation = kb.node();
      kb.new Invocation(invocation, kb.node(BuiltIn.print)).exceptionHandler(exceptionHandler);
      invocation.activate(new Context(kb::node));

      val activation = monitor.emissions().blockingFirst();
      final Node exception = activation.context.node.properties.get(kb.node(Common.exception));
      // TODO(rosswang): Once we support node-space stack traces, the deepest frame in
      // this case may be BuiltIn.print, followed by invocation.
      assertSame(invocation, exception.properties.get(kb.node(Common.source)));
    }
  }
}
