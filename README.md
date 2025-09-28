# DataVisGen

An intelligent data visualization web application that automatically generates charts and graphs from CSV and Excel files using natural language prompts and AI-powered analysis.

## Features

- **AI-Powered Visualization**: Uses Mistral AI to interpret natural language prompts and generate appropriate visualizations
- **Multi-Format Support**: Accepts CSV, XLSX, and XLS files
- **Interactive Web Interface**: Clean, responsive web interface with drag-and-drop file upload
- **Automatic Chart Generation**: Creates histograms, timelines, scatter plots, bar charts, and tables
- **Refinement Capabilities**: Iteratively improve visualizations with follow-up prompts
- **Auto-Report Generation**: AI suggests relevant visualizations based on your dataset
- **Multi-Language Support**: Interface available in French, English, Spanish, Italian, and German
- **Export Functionality**: Download generated charts as PNG files

## Technology Stack

- **Backend**: Python Flask
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **AI Integration**: Mistral API
- **Frontend**: HTML5, CSS3, JavaScript
- **File Handling**: Werkzeug

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DataVisGen
```

2. Install Python dependencies:
```bash
pip install flask pandas numpy matplotlib werkzeug
```

3. Set up your Mistral API key:
```bash
export MISTRAL_API_KEY="your-mistral-api-key-here"
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:8000`

## Usage

1. **Upload Data**: Drag and drop or select a CSV/Excel file
2. **Enter Prompt**: Describe what visualizations you want (e.g., "Show sales trends over time", "Create a histogram of customer ages")
3. **Generate**: Click "Generate" to create visualizations
4. **Refine**: Use the refinement feature to modify existing charts
5. **Auto-Report**: Let AI suggest relevant visualizations for your dataset

## Example Prompts

- "Show a timeline of appointments by type"
- "Create a histogram of ingredient counts with 20 bins"
- "Display a table of the top 20 records"
- "Generate a scatter plot of price vs. quantity"
- "Show bar chart of sales by category"

## API Configuration

The application uses Mistral AI for natural language processing. You can configure the API endpoint and model in the code:

- Default model: `mistral-large-latest`
- Default endpoint: `https://api.mistral.ai/v1/chat/completions`

## File Structure

```
DataVisGen/
├── app.py              # Flask web application
├── autovis.py          # Core visualization engine
├── static/
│   ├── plots/          # Generated chart images
│   └── style.css       # Application styling
├── templates/
│   └── index.html      # Main web interface
└── uploads/            # Temporary file storage
```
