
from typing import Any
import threading
import textwrap


class MyiaSyntaxError(Exception):
    """
    Class for syntax errors in Myia. This exception type should be
    raised for any feature that is not supported.

    Attributes:
        location: The error's location in the original source.
        message: A precise assessment of the problem.
    """
    def __init__(self, message: str, location: 'Location' = None) -> None:
        self.location = location or current_location()
        self.message = message
        super().__init__(self.message, self.location)


_about = threading.local()
_about.stack = [None]


def top():
    return _about.stack[-1]


def current_location():
    abt = top()
    while abt:
        node = abt.node
        if getattr(node, 'about', None):
            abt = node.about
        else:
            abt = None
    if isinstance(node, Location):
        return node
    return None


class About:
    def __init__(self, node, transform):
        self.node = node
        self.transform = transform

    def __enter__(self):
        _about.stack.append(self)

    def __exit__(self, etype, evalue, etraceback):
        _about.stack.pop()


class AboutPrinter:
    def __init__(self, node):
        self.node = node

    def node_hrepr(self, node, H, hrepr):
        if hasattr(node, 'trace'):
            views = H.tabbedView()
            views = views(H.view(H.tab('node'), H.pane(hrepr(node))))
            views = views(H.view(H.tab('trace'), H.pane(hrepr(node.trace))))
            return views
        else:
            return hrepr(node)

    def __hrepr__(self, H, hrepr):
        views = H.tabbedView()
        node = self.node

        nodes = [self.node_hrepr(node, H, hrepr)]
        transforms = []

        while node and getattr(node, 'about', None):
            about = node.about
            node = about.node
            nodes.append(self.node_hrepr(node, H, hrepr))
            transforms.append(about.transform)
        transforms.append('orig')

        for transform, node in reversed(list(zip(transforms, nodes))):
            tab = H.tab(transform)
            pane = H.pane(node)
            views = views(H.view(tab, pane))

        return views


class Location:
    """
    Represents a source code location for an AST node.

    Attributes:
        url (str): The path of the code file.
        line (int): The line number in that file.
        column (int): The column number in that file.
    """

    def __init__(self,
                 url: str,
                 line: int,
                 column: int,
                 node: Any = None) -> None:
        self.url = url
        self.line = line
        self.column = column
        self.node = node

    def traceback(self) -> str:
        """
        Print out a "traceback" that corresponds to this location,
        with the line printed out and a caret at the right column.
        Basically:

        >>> loc.traceback()
          File {url}, line {line}, column {column}
            x = f(y)
                ^

        This is mostly meant for printing out ``MyiaSyntaxError``s.
        """
        try:
            with open(self.url) as file:
                raw_code = file.readlines()[self.line - 1].rstrip("\n")
                raw_caret = ' ' * self.column + '^'
                code, caret = textwrap.dedent(
                    raw_code + '\n' + raw_caret
                ).split("\n")
            return '  File "{}", line {}, column {}\n    {}\n    {}'.format(
                self.url, self.line, self.column, code, caret)
        except FileNotFoundError:
            return '  File "{}", line {}, column {}'.format(
                self.url, self.line, self.column)

    def __str__(self) -> str:
        return '{}@{}:{}'.format(self.url, self.line, self.column)

    def __hrepr__(self, H, hrepr):
        return H.codeSnippet(
            src = self.url,
            language = "python",
            line = self.line,
            column = self.column + 1,
            context = 4
        )
