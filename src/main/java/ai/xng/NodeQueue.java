package ai.xng;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.lang.ref.WeakReference;
import java.util.Collection;
import java.util.ConcurrentModificationException;
import java.util.Iterator;
import java.util.NoSuchElementException;

import com.google.common.collect.Iterables;

import io.reactivex.Observable;
import io.reactivex.disposables.CompositeDisposable;
import io.reactivex.subjects.PublishSubject;
import io.reactivex.subjects.Subject;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;

/**
 * A collection of nodes ordered by most recent activation. This collection is
 * not thread-safe.
 * 
 * This collection used to hold weak references to its nodes. This behavior may
 * be reintroduced in the future.
 */
public class NodeQueue implements Iterable<Node>, Serializable {
  private static final long serialVersionUID = -2635392533122747827L;

  @RequiredArgsConstructor
  private class Link {
    Link previous, next;
    @NonNull
    final Node node;
  }

  private class NodeQueueIterator implements Iterator<Node> {
    final Object version;
    Link nextLink;

    NodeQueueIterator() {
      version = NodeQueue.this.version;
      nextLink = head;
    }

    @Override
    public boolean hasNext() {
      return nextLink != null;
    }

    @Override
    public Node next() {
      if (version != NodeQueue.this.version) {
        throw new ConcurrentModificationException();
      }

      if (!hasNext())
        throw new NoSuchElementException();

      final Node nextNode = nextLink.node;
      nextLink = nextLink.next;
      return nextNode;
    }
  }

  private transient CompositeDisposable contextDisposable;
  private transient WeakReference<Context> context;
  private transient Link head, tail;
  private transient Object version;

  private transient Subject<Node> rxActivate;

  public Observable<Node> rxActivate() {
    return rxActivate;
  }

  public NodeQueue(final Context context) {
    init(context);
  }

  private void init(final Context context) {
    if (context != null) {
      contextDisposable = new CompositeDisposable();
      this.context = new DisposingWeakReference<>(context, contextDisposable);
    }
    rxActivate = PublishSubject.create();
  }

  private void writeObject(final ObjectOutputStream o) throws IOException {
    o.defaultWriteObject();
    if (context == null) {
      o.writeBoolean(false);
    } else {
      o.writeBoolean(true);
      o.writeObject(context.get());
    }

    for (final Node node : this) {
      o.writeObject(node);
    }
    o.writeObject(null);
  }

  private void readObject(final ObjectInputStream stream) throws ClassNotFoundException, IOException {
    stream.defaultReadObject();
    init(stream.readBoolean() ? (Context) stream.readObject() : null);
    Node node;
    while ((node = (Node) stream.readObject()) != null) {
      add(node);
    }
  }

  public void add(final Node node) {
    final Link link = new Link(node);
    initAtTail(link);

    if (context == null) {
      node.rxActivate().subscribe(a -> {
        promote(link);
        rxActivate.onNext(node);
      });
    } else {
      contextDisposable.add(node.rxActivate().subscribe(a -> {
        if (a.context == context.get()) {
          promote(link);
          rxActivate.onNext(node);
        }
      }));
    }
  }

  public void addAll(final Collection<Node> nodes) {
    for (final Node node : nodes) {
      add(node);
    }
  }

  private void remove(final Link link) {
    if (link.previous == null) {
      head = link.next;
    } else {
      link.previous.next = link.next;
    }

    if (link.next == null) {
      tail = link.previous;
    } else {
      link.next.previous = link.previous;
    }

    version = null;
  }

  private void initAtTail(final Link link) {
    if (tail == null) {
      head = tail = link;
    } else {
      link.previous = tail;
      tail.next = link;
      tail = link;
    }

    version = null;
  }

  private void promote(final Link link) {
    if (link == head)
      return;

    remove(link);

    link.previous = null;
    link.next = head;
    head.previous = link;
    head = link;

    version = null;
  }

  @Override
  public Iterator<Node> iterator() {
    if (version == null) {
      version = new Object();
    }
    return new NodeQueueIterator();
  }

  @Override
  public String toString() {
    return Iterables.toString(this);
  }
}
