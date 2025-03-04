# Identity Disambiguation System Architecture

## System Overview

This document outlines a streamlined architecture for a system that disambiguates between two individuals named Samuel Jackson - one with legitimate activities ("good Samuel") and another involved in money laundering ("bad Samuel"). The system analyzes articles, builds a graph representation, and classifies new articles based on graph features.

## Simplified Architecture

The system follows a consolidated design with related functionality grouped together:

```
samuel-identity-system/
├── app.py                 # Main Streamlit application
├── article_processor.py   # Article scraping and text processing
├── graph_manager.py       # Graph construction and analysis
├── classifier.py          # Similarity calculation and classification
├── visualization.py       # Graph and result visualization
├── utils.py               # Utility functions
├── config.py              # Configuration settings
├── tests/                 # Testing suite
│   ├── test_app.py        # Application tests
│   ├── test_graph.py      # Graph tests
│   └── test_classifier.py # Classifier tests
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Component Details

### app.py

The main Streamlit application that ties everything together.

**Key Functions:**
```python
def main():
    """Main application entry point"""
    
def process_article_url(url):
    """Process a new article URL"""
    
def display_results(article_id, classification_result):
    """Display classification results"""
```

### article_processor.py

Handles article scraping, content extraction, and text processing.

**Key Functions:**
```python
def extract_article_content(url):
    """Extract content from an article URL"""
    
def preprocess_text(text):
    """Preprocess article text for analysis"""
    
def extract_entities(text):
    """Extract named entities from article text"""
    
class Article:
    """Class representing an article with all its properties"""
```

### graph_manager.py

Manages the article graph, including construction, updates, and analysis.

**Key Functions:**
```python
class ArticleGraph:
    """Manages the article graph and provides analysis methods"""
    
    def __init__(self):
        """Initialize the graph"""
        
    def add_article(self, article):
        """Add an article to the graph"""
        
    def calculate_similarities(self, article_id):
        """Calculate similarities between an article and all others"""
        
    def update_edges(self, article_id):
        """Update graph edges based on similarities"""
        
    def extract_features(self, article_id):
        """Extract graph features for classification"""
        
    def save_graph(self, filepath):
        """Save the graph to a file"""
        
    def load_graph(self, filepath):
        """Load a graph from a file"""
```

### classifier.py

Handles similarity calculation and classification of articles.

**Key Functions:**
```python
def calculate_text_similarity(text1, text2):
    """Calculate similarity between two text documents"""
    
def calculate_contextual_similarity(article1, article2):
    """Calculate similarity based on contextual factors"""
    
class GraphClassifier:
    """Classifies articles based on graph features"""
    
    def train(self, graph, labeled_nodes):
        """Train the classifier using labeled nodes"""
        
    def predict(self, features):
        """Predict cluster for an article based on features"""
        
    def evaluate(self, graph, test_nodes):
        """Evaluate classifier performance"""
```

### visualization.py

Handles graph visualization and result display.

**Key Functions:**
```python
def create_graph_visualization(graph, highlight_node=None):
    """Create a visualization of the graph"""
    
def display_classification_result(result, confidence):
    """Display classification result with confidence"""
    
def display_evidence(supporting_nodes, weights):
    """Show evidence supporting the classification"""
```

### utils.py

Contains utility functions used across the system.

**Key Functions:**
```python
def validate_url(url):
    """Validate that a URL is properly formatted"""
    
def load_config(config_file):
    """Load configuration from file"""
    
def setup_logging():
    """Set up logging configuration"""
```

### config.py

Contains configuration settings for the application.

**Key Variables:**
```python
# Scraping configuration
USER_AGENT = "Mozilla/5.0 ..."
REQUEST_TIMEOUT = 10

# Graph configuration
SIMILARITY_THRESHOLD = 0.3
EDGE_WEIGHT_MULTIPLIER = 1.0

# Classifier configuration
TRAIN_TEST_SPLIT = 0.8
CROSS_VALIDATION_FOLDS = 5

# UI configuration
GRAPH_SIZE = (800, 600)
COLOR_MAP = {
    "good": "#2ECC71",  # Green
    "bad": "#E74C3C",   # Red
    "unknown": "#95A5A6"  # Gray
}
```

## Data Flow

```
User Input (URL) → article_processor.py → graph_manager.py → classifier.py → visualization.py → User Interface
```

## Core Workflows

### Processing a New Article

```python
# In app.py
def process_article_url(url):
    # Extract and preprocess article
    article = article_processor.extract_article_content(url)
    
    # Add to graph
    graph.add_article(article)
    
    # Update edges
    graph.update_edges(article.id)
    
    # Extract features
    features = graph.extract_features(article.id)
    
    # Classify
    result = classifier.predict(features)
    
    # Display results
    display_results(article.id, result)
    return article.id, result
```

### Training the Classifier

```python
# In app.py or a separate script
def train_classifier():
    # Get labeled articles
    labeled_articles = graph.get_labeled_nodes()
    
    # Split into train/test
    train_nodes, test_nodes = split_train_test(labeled_articles)
    
    # Train classifier
    classifier.train(graph, train_nodes)
    
    # Evaluate performance
    performance = classifier.evaluate(graph, test_nodes)
    
    return performance
```

## Implementation Approach

### Phase 1: Core Functionality
1. Implement basic article extraction in `article_processor.py`
2. Create the graph structure in `graph_manager.py`
3. Implement simple text similarity in `classifier.py`
4. Set up basic visualization in `visualization.py`
5. Create basic Streamlit UI in `app.py`

### Phase 2: Enhanced Features
1. Improve article extraction with better entity recognition
2. Implement more sophisticated graph features
3. Enhance the classifier with better algorithms
4. Improve visualizations with interactive elements

### Phase 3: Refinement
1. Add model persistence (save/load)
2. Implement performance optimizations
3. Add detailed explanations for classifications
4. Create comprehensive testing suite

## Key Technologies

1. **Framework**: Streamlit for UI
2. **Graph Library**: NetworkX for graph operations
3. **NLP**: spaCy for entity extraction, Sentence Transformers for embeddings
4. **Visualization**: Matplotlib/Plotly for graph visualization
5. **Machine Learning**: scikit-learn for classification

## Data Structures

### Article
```python
class Article:
    def __init__(self, url, title, content, publication_date=None, source=None):
        self.id = generate_id()  # Unique identifier
        self.url = url
        self.title = title
        self.content = content
        self.publication_date = publication_date
        self.source = source
        self.entities = {}  # Named entities extracted from content
        self.cluster = None  # "good", "bad", or None if unknown
```

### Classification Result
```python
class ClassificationResult:
    def __init__(self, article_id, predicted_cluster, confidence, supporting_evidence=None):
        self.article_id = article_id
        self.predicted_cluster = predicted_cluster  # "good" or "bad"
        self.confidence = confidence  # 0.0 to 1.0
        self.supporting_evidence = supporting_evidence or []  # List of supporting nodes
```
<!-- ## Future Extensions

1. **Batch Processing**: Add capability to process multiple articles at once
2. **Active Learning**: Implement feedback mechanism to improve the model over time
3. **Advanced Visualization**: Add more interactive visualization options
4. **API Interface**: Create simple API for integration with other systems -->
Testing articles 
- Bad: https://www.cbsnews.com/chicago/news/3-chicago-area-residents-charged-with-covid-19-relief-fraud/ 
- Good: https://movieweb.com/samuel-l-jackson-robert-downey-jr-reunite-oscars/