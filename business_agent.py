from openai import OpenAI
from pydantic import BaseModel
from typing import Literal, List
import os

try:
    from google.colab import userdata
    PINECONE_API_KEY = userdata.get('PINECONE_API_KEY')
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    OPENAI_BASE = userdata.get('OPENAI_BASE', 'https://47v4us7kyypinfb5lcligtc3x40ygqbs.lambda-url.us-east-1.on.aws/v1/')
except:
    from dotenv import load_dotenv
    load_dotenv('.env')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE = os.getenv('OPENAI_BASE', 'https://47v4us7kyypinfb5lcligtc3x40ygqbs.lambda-url.us-east-1.on.aws/v1/')


# Define structured output schemas using Pydantic
class EmailClassification(BaseModel):
    """Schema for email classification results"""
    category: Literal["product inquiry", "order request"]
    confidence: float  # Between 0.0 and 1.0
    reasoning: str     # Brief explanation of the classification


class OrderItem(BaseModel):
    """Schema for individual order items"""
    product_mentioned: str    # Product name/description mentioned by customer
    product_id: str          # Matched product ID from catalog
    quantity: int            # Quantity requested (default 1 if not specified)


class OrderAnalysis(BaseModel):
    """Schema for analyzing order requests"""
    order_items: List[OrderItem]  # List of ordered items
    customer_intent: str          # What the customer wants to do
    urgency_level: Literal["low", "medium", "high"]


class BusinessAgent:
    def __init__(self, client, vector_store):
        self.client = client
        self.vector_store = vector_store

    def classify_email_structured(self, email_subject: str, email_text: str, model: str = "gpt-4o") -> EmailClassification:
        """
        Classify an email using structured output to ensure consistent format
        """
        system_prompt = """
        You are an AI assistant helping classify customer emails for a fashion store.
        
        Your task is to analyze emails and classify them as either:
        - "product inquiry": Customer asking questions about products, availability, features, etc.
        - "order request": Customer wanting to purchase, order, or buy specific items
        
        Provide a confidence score between 0.0 and 1.0 and explain your reasoning briefly.
        """
        
        response = self.client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please classify this email:\n\nSubject: {email_subject}\n\nBody: {email_text}"}
            ],
            response_format=EmailClassification
        )
        
        return response.choices[0].message.parsed
    
    def analyze_order_request(self, email_subject: str, email_text: str, model: str = "gpt-4o") -> OrderAnalysis:
        """
        Analyze an order request email to extract structured product information
        """
        system_prompt = """
        You are analyzing order request emails for a fashion store.
        Extract the following information for each product mentioned:
        - Create an OrderItem for each product the customer wants to order
        - For each OrderItem, include:
          * product_mentioned: The exact product name/description the customer used
          * product_id: The matching product ID from the catalog below
          * quantity: How many items (use 1 if not specified)

        Also determine:
        - customer_intent: What the customer wants to do overall
        - urgency_level: Based on language used (low/medium/high)

        Use the product catalog below to find the most relevant product ID for each item:

        {retrieved_context}
        
        Make sure to match products based on name, category, season, and description.
        If multiple products are mentioned, create separate OrderItem entries for each.
        """

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        retrieved_docs = retriever.invoke(f"{email_subject}\n\n{email_text}")

        # Format the retrieved context to include metadata fields
        context_parts = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            context_part = f"""
Product ID: {metadata.get('product_id', 'N/A')}
Name: {metadata.get('name', 'N/A')}
Category: {metadata.get('category', 'N/A')}
Seasons: {metadata.get('seasons', 'N/A')}
Description: {doc.page_content}
---"""
            context_parts.append(context_part)
        
        retrieved_context = "\n".join(context_parts)
        response = self.client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.format(retrieved_context=retrieved_context)},
                {"role": "user", "content": f"Analyze this order request:\n\nSubject: {email_subject}\n\nBody: {email_text}"}
            ],
            response_format=OrderAnalysis
        )
        
        return response.choices[0].message.parsed
    
    def classify_emails(self, email_text, model="gpt-4o"):
        """Legacy method - kept for backwards compatibility"""
        result = self.classify_email_structured(email_text, model)
        return result.category


# Example usage
if __name__ == "__main__":
    # Initialize OpenAI client with custom base URL
    client = OpenAI(
        base_url=OPENAI_BASE,
        api_key=OPENAI_API_KEY
    )

    # Initialize vector store (assuming you have a VectorstoreFactory class)
    from vectorstore import VectorstoreFactory
    from pinecone import Pinecone
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    vector_store_factory = VectorstoreFactory(pinecone)
    vector_store = vector_store_factory.get_vectorstore('products')

    # Create the business agent
    agent = BusinessAgent(client, vector_store)
    
    # Example email texts
    sample_inquiry = """
    Hi there! I'm looking for a nice summer dress for a wedding. 
    Do you have any floral patterns available in size M? 
    What's the price range for your formal dresses?
    """
    
    sample_order = """
    Hello, I'd like to order 2 blue denim jackets in size L and 1 red scarf. 
    Please let me know if they're available and how much it will cost.
    I need them by next Friday for a trip.
    """
    
    # Classify emails with structured output
    print("=== Email Classification ===")
    
    inquiry_result = agent.classify_email_structured("Summer dress inquiry", sample_inquiry)
    print(f"Email 1 Classification:")
    print(f"  Category: {inquiry_result.category}")
    print(f"  Confidence: {inquiry_result.confidence:.2f}")
    print(f"  Reasoning: {inquiry_result.reasoning}")
    print()
    
    order_result = agent.classify_email_structured("Urgent order request", sample_order)
    print(f"Email 2 Classification:")
    print(f"  Category: {order_result.category}")
    print(f"  Confidence: {order_result.confidence:.2f}")
    print(f"  Reasoning: {order_result.reasoning}")
    print()
    
    # Analyze order request in detail
    if order_result.category == "order request":
        print("=== Order Analysis ===")
        subject = "Urgent order request"
        order_analysis = agent.analyze_order_request(subject, sample_order)
        
        print(f"Customer intent: {order_analysis.customer_intent}")
        print(f"Urgency level: {order_analysis.urgency_level}")
        print(f"Order items ({len(order_analysis.order_items)}):")
        
        for i, item in enumerate(order_analysis.order_items, 1):
            print(f"  Item {i}:")
            print(f"    Product mentioned: {item.product_mentioned}")
            print(f"    Product ID: {item.product_id}")
            print(f"    Quantity: {item.quantity}")
