# CoLLT
Contrastive Learning for Long Document Transformers 
To do:
1. Final Report
  * Introduction - Copy from proposal
  * Related Work - Copy from proposal, Cite 10 papers, only 6 so far
  * Baselines - Nidhi
  * Datasets - Alok ( Add a histogram of text length distribution)
  * Approach - Siddhant and Mehek
  * Add barlow twins explanation - Siddhant
  * Experiment 1 results
  * Experiment 2 results
  * Experiment 3 results
  * Error analysis
  * Conclusion and Future work
2. Experiment 1
  * Baseline 1 - Run BERT on IMDB 1000 subset, just fine tune - Alok
  * Baseline 2 - Run BERT on IMDB 1000 subset, model tune and fine tune - Alok
  * Baseline 3 - Run longformer on IMDB 1000 subset, model tune and fine tune - Nidhi
  * Contrastive on BERT, model tune and fine tune - Nidhi, Shingi, Alok
  * Contrastive on BERT, just fine tune - Shingi, Nidhi, Alok
3. Experiment 2
  * Contrastive on longformer, model tune and fine tune - Shingi, Alok, Nidhi
4. Experiment 3
  * Trying different augmentation types. (Random chunk, non overlapping chunk , overlapping chunk) - Mehek
5. Update Codebase
  * Add baseline files
  * Add augmentation stuff
  * Generalise augmentation in model
