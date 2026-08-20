from markdown_it import MarkdownIt
from bs4 import BeautifulSoup
import re

md = MarkdownIt("commonmark").enable(["table", "strikethrough", "linkify"])

x = BeautifulSoup(
    md.render("""
# This is a heading

[asfd](https://example.com)

## a simple table

a | b
--|--
`code` | `code`
"""),
    "html.parser",
)

x = x.get_text().strip()

x = re.sub(r" *\n\s*", "\n", x)
print(x)
