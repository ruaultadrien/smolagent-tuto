from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_documents():
    # Simulate a knowledge base about party planning
    party_ideas = [
        {
            "text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.",
            "source": "Party Ideas 1",
        },
        {
            "text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.",
            "source": "Entertainment Ideas",
        },
        {
            "text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'",
            "source": "Catering Ideas",
        },
        {
            "text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.",
            "source": "Decoration Ideas",
        },
        {
            "text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.",
            "source": "Entertainment Ideas",
        },
    ]

    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in party_ideas
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return text_splitter.split_documents(source_docs)
