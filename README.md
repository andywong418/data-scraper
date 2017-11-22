# data-scraper
I trained a web scraper from a validation set (of cats and dogs in this case) using a computationally cheap method such as K-nearest neighbours to scrape only images that lead to higher binary classification accuracies into an input dataset.
Then I improve the accuracy of web scraper by slicing the last two layers of a pre-trained convolutional neural network (VGG ImageNet) and re-training the last layers with our curated input dataset.
Achieved 93.4% test classification accuracy based on 5000 scraped images of cats, dogs (and a neither class).
