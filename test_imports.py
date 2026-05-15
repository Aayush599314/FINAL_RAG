
try:
    from langchain.chains import RetrievalQA
    print("Found in langchain.chains")
except ImportError:
    print("Not found in langchain.chains")

try:
    from langchain_community.chains import RetrievalQA
    print("Found in langchain_community.chains")
except ImportError:
    print("Not found in langchain_community.chains")

try:
    from langchain_classic.chains import RetrievalQA
    print("Found in langchain_classic.chains")
except ImportError:
    print("Not found in langchain_classic.chains")
