from summarizer import Summarizer
from rouge_score import rouge_scorer

def calculate_rouge(reference_text, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_summary)

    rouge2_score = scores['rouge2']
    
    return rouge2_score

with open("text.txt", 'r', encoding="UTF-8") as file:
    data = file.read().replace('\n', ' ')

model = Summarizer()
sazetak = model(data, num_sentences=3, min_length=20)
sazetak = str(sazetak)

print("Generated Summary:")
print(sazetak)

reference_summary = """In a mystical forest known as "The Enchanted Forest," sunlight barely touches the moss-covered ground, creating a world of magic where reality seems to blur. Majestic trees with golden leaves and a fragrant air filled with wildflowers and birdsongs define the forest. Legends speak of fairies, unicorns, and ancient spirits inhabiting this magical place. Eliza, a young adventurer, enters the forest with a sketchbook, hoping to uncover its magic. She discovers a stream and a glade where fairies gather, sketching their beauty and grace. As twilight falls, Eliza leaves the forest, her heart full of joy and wonder, and a sketchbook full of magical memories."""

rouge2_score = calculate_rouge(reference_summary, sazetak)

print(f"ROUGE-2 Score: Precision: {rouge2_score.precision:.4f}, Recall: {rouge2_score.recall:.4f}, F1 Score: {rouge2_score.fmeasure:.4f}")
