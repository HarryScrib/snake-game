import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot(scores, mean_scores):
    plt.figure(figsize=(12, 8)) # create new figure
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, color='#FF9999', label='Score')  
    plt.plot(mean_scores, color='#CC99FF', label='Mean Score') 
    plt.ylim(ymin=0)
    plt.legend()
    
    # add text labels for latest values
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        
    # save plot with timestamp to avoid overwriting
    if not os.path.exists('./model'):
        os.makedirs('./model')
        
    # create unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'./model/training_results_{timestamp}.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Training plot saved to {filename}")
    
    plt.show