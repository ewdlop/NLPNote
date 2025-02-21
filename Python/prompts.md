# prompts

```txt
If I have two documents and they are only differ by timestamp it few characters, what should I do to aggregate them?
```

```python
import difflib

def compare_documents(doc1, doc2):
    d = difflib.Differ()
    diff = list(d.compare(doc1.splitlines(), doc2.splitlines()))
    
    print("\n".join(diff))

# Example documents with only timestamp differences
doc1 = """Order ID: 12345
Customer: John Doe
Timestamp: 2025-02-21 10:00:00"""

doc2 = """Order ID: 12345
Customer: John Doe
Timestamp: 2025-02-21 10:05:00"""

compare_documents(doc1, doc2)

```

```python

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

doc1 = """Order ID: 12345
Customer: John Doe
Timestamp: 2025-02-21 10:00:00"""

doc2 = """Order ID: 12345
Customer: John Doe
Timestamp: 2025-02-21 10:05:00"""

similarity = fuzz.ratio(doc1, doc2)

print(f"Document Similarity: {similarity}%")

```
