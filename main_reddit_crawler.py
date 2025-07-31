import praw
import pymysql
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForTokenClassification, pipeline , BartTokenizer, BartForConditionalGeneration
import psutil
from rake_nltk import Rake
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.disable(logging.CRITICAL)
import time 
import json
from langdetect import detect
import re
import asyncio
import os
import math
from aiohttp import ClientSession

# Function to get CPU and memory usage
def get_system_stats():
    # CPU usage as a percentage
    cpu_usage = psutil.cpu_percent(interval=1)  # You can adjust the interval as needed

    # Memory usage
    memory = psutil.virtual_memory()
    memory_usage = memory.percent

    return cpu_usage, memory_usage






# Token bucket rate limiting class
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill_time = time.time()

    async def consume(self):
        while True:
            current_time = time.time()
            time_elapsed = current_time - self.last_refill_time

            # Refill tokens if needed
            if time_elapsed > 1.0:
                tokens_to_add = int(time_elapsed * self.rate)
                self.tokens = min(self.capacity, self.tokens + tokens_to_add)
                self.last_refill_time = current_time

            if self.tokens >= 1:
                self.tokens -= 1
                return True  # Token available, allow the call
            else:
                await asyncio.sleep(0.1)  # Sleep briefly and check again




# Function to get CPU and memory usage
def get_system_stats():
    # CPU usage as a percentage
    cpu_usage = psutil.cpu_percent(interval=1)  # You can adjust the interval as needed

    # Memory usage
    memory = psutil.virtual_memory()
    memory_usage = memory.percent

    return cpu_usage, memory_usage


#summerize post_content using BART model
def summarize(post_content):
    
    word_count = len(post_content.split())
    # Load pre-trained BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    print("summarize")
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.to("cuda:0")  # Specify the device here, e.g., 
        

    # Tokenize and generate summary
    input_ids = tokenizer.encode(post_content, return_tensors="pt", max_length=700, truncation=True).to("cuda:0")
    summary_ids = model.generate(input_ids, max_length=100, num_beams=4, length_penalty=2.0, early_stopping=True)

    # Decode the generated summary
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    
    print(post_content)
    
    if word_count > 199:
        max_length_162_percent = int(word_count * 0.939)
        max_length_37_percent = int(word_count * 0.639)
        
        return summary_text  
    else:
        return post_content
    
   




# Function to find and split sentences in a text based on punctuation marks
def find_and_split_sentences(text):
    sentence_endings = re.finditer(r'[.!?;|)]\s+', text)

    for x in sentence_endings:
        print(x,"endigns")
    sentence_endings_indexes = [match.start() for match in sentence_endings]
    print(sentence_endings_indexes)
    if sentence_endings_indexes == []:
        sentence_endings_indexes.append(len(text) - 1)
    
    # Check for closely spaced punctuation marks
    closely_spaced_indexes = []
    for i in range(1, len(sentence_endings_indexes)):
        if sentence_endings_indexes[i] - sentence_endings_indexes[i - 1] <= 2:
            closely_spaced_indexes.append(sentence_endings_indexes[i - 1])
            closely_spaced_indexes.append(sentence_endings_indexes[i])
    
    # Combine closely spaced punctuation marks into sentences
    sentences = []
    char_index = 0
    lengths = []
    sentence_counter = 0 

    for idx, end_index in enumerate(sentence_endings_indexes):
        sentence = text[char_index:end_index + 1].strip()
        sentence_counter += 1 
        print(sentence_counter)
        print("len sentence",len(sentence))
        
        if sentence.endswith("'s") and idx + 1 < len(sentence_endings_indexes):
            next_sentence_start = sentence_endings_indexes[idx + 1] + 1
            next_sentence = text[end_index + 1:next_sentence_start].strip()
            print("''''''''''''")
            sentence += " " + next_sentence
            char_index = next_sentence_start
        else:
            print(end_index)
            char_index = end_index + 1

        l = len(sentence)
        print("l = = = = = ", l)
        
        if l == 0:
            print(00000000)
            l = 1
        lengths.append(l)
        sentences.append(sentence)
        
        # Check if the current index is a part of closely spaced punctuation
        if end_index in closely_spaced_indexes:
            char_index = end_index + 1

    if char_index < len(text):
        sentences.append(text[char_index:].strip())
        sentence_counter += 1  # Increment the counter when appending the last sentence

    return sentences, lengths, sentence_counter








## Function to get entities from a batch of input sentences using BERT NER model
def get_entities_batch(input_sentences):
    
    tokenizer = BertTokenizer.from_pretrained("dslim/bert-large-NER")
    model = BertForTokenClassification.from_pretrained("dslim/bert-large-NER")
    print("get ents")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0) 
    print("------------------")
    ner_results_list = nlp(input_sentences)
    formatted_entities_batch = []

    for ner_results in ner_results_list:
        formatted_entities = []
        formatted_categories = []
        current_entity = ''
        current_word = ''

        for result in ner_results:
            entity_type = result['entity'][2:]  # Get the 3-letter entity category
            word = result['word']
            addSpace = False
            if all(val in word for val in '##'):
                word = result['word'].replace('##', '')
                print("replaced")
            else:
                addSpace = True
            
            if result['entity'].startswith('B-'):
                if current_entity and current_word:
                    formatted_categories.append(current_entity)
                    formatted_entities.append(current_word)
                current_entity = entity_type
                current_word = word
            elif result['entity'].startswith('I-'):
                if addSpace:
                    current_word += ' ' + word
                else:
                    current_word += word

            else:
                continue
        
        if current_entity and current_word:
            formatted_categories.append(current_entity)
            formatted_entities.append(current_word)
        
        combined_entities = []
        combined_categories = []
        i = 0
        while i < len(formatted_entities):
            entity = formatted_entities[i]
            category = formatted_categories[i]
            
            if i < len(formatted_entities) - 1 and formatted_categories[i] == formatted_categories[i + 1]:
                i += 1
                while i < len(formatted_entities) and formatted_categories[i] == 'I-' + category:
                    entity += formatted_entities[i]
                    i += 1
            else:
                i += 1
            
            combined_entities.append(entity)
            combined_categories.append(category)
        
        formatted_entities_batch.append((combined_entities, combined_categories))
    print("------------------")
    return formatted_entities_batch

 # Function to get entities and sentiment from input sentences
def get_entities(input_sentences):
    sentiment = getSentiment(input_sentences)
    print("-")
    sentences, lengths, sentence_counter = find_and_split_sentences(input_sentences)
    print("-")
    print(sentence_counter)
    print("-")
    print("lens", lengths)
    print("len sentences",len(sentences))
    print(sentences)
    
    for index, sentence in enumerate(sentences):
        
        # Your processing logic for each sentence
        # 'index' will hold the current index of the sentence
        # 'sentence' will hold the current sentence in each iteration
        # Add your code here to process 'sentence' and use 'index' if needed
        print("Index:", index,"==================")
        
        print("sentence len", len(sentence))

        

        
        print("lengths len", len(lengths))
        print(lengths, "lengths")
        print(lengths[index])
        batch_size = (lengths[index] )  # You can adjust this based on memory and performance
        print("b size",batch_size)
        input_batches = [input_sentences[i:i+batch_size] for i in range(0, len(input_sentences), batch_size)]
        
        formatted_entities_batch = get_entities_batch(input_batches)
          # Flatten the results and return
        formatted_entities = []
        formatted_categories = []
        print("------------------")
        for entities, categories in formatted_entities_batch:
            print("ents", entities)
            print("cats", categories)
            formatted_entities.extend(entities)
            formatted_categories.extend(categories)
            
            

        
    
    combined_entsWithCatagories = list(zip(formatted_entities, formatted_categories))

        
    print("ents are ", formatted_entities)
    print("cats are ", formatted_categories)
    print("sents are ", sentiment)
    return combined_entsWithCatagories, sentiment






# Function to get sentiment analysis of a given text using a pre-trained model
def getSentiment(post_content): 
    
        print("get sent")
        pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)  # Use 'device=0' for the first GPU
        
        # Perform sentiment analysis on the input text "sentences"
        results = pipe(post_content)  
        
        for x in results:
            print("sentiment--", x)
       
        return results





    


    ## Function to remove URLs from a given text
def remove_urls(text):
    print(type(text))
    # Define a pattern to match URLs
    url_pattern = re.compile(r'\b(?:https?://|www\.)\S+\b')
    
    # Replace URLs with an empty string
    text_without_urls = re.sub(url_pattern, '', text)
    print(text_without_urls, "no urls")
    return text_without_urls




def rake(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords_with_scores = r.get_ranked_phrases_with_scores()  # Updated method
    
    return keywords_with_scores


def crawl_get_values(subreddit_name):
        reddit = define_reddit()
        
        subreddit = reddit.subreddit(subreddit_name)

        # Get new posts from the subreddit
        num_posts = 5 # Adjust the number of posts you want to retrieve
        for submission in subreddit.new(limit=num_posts):
            submission_attributes = vars(submission)
            for attribute, value in submission_attributes.items():
                print(f"{attribute}: {value}")
         

def define_reddit():
    user=os.environ.get("reddit_user"),
    reddit = praw.Reddit(
        client_id=os.environ.get("reddit_client_id"),
        client_secret=os.environ.get("reddit_client_secret"),
        
        user_agent=f'windows:Crawler:v1.00 by{user} ',
    )
    return reddit

def count_posts_from_three_days_ago(subreddit_name):
    print("start count_")
    reddit = define_reddit()
    subreddit = reddit.subreddit(subreddit_name)
    
    # Get the current time   
     

    # Calculate the timestamp for the start of 3 days ago
    three_days_ago = datetime.now() - timedelta(days=0)
  
    start_of_three_days_ago = three_days_ago.replace(hour=0, minute=0, second=0, microsecond=0)
    print("get timestamp")
    timestamp = int(start_of_three_days_ago.timestamp())
# Convert to a Unix timestamp (if needed).timestamp()))
    print(start_of_three_days_ago)
    count = 0
   
    for submission in subreddit.new(limit=None):
        
        if submission.created_utc < timestamp:
            break
        count += 1

    print("returned count from time",count)
    return count,  start_of_three_days_ago

def is_image_url(url):
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', 'gifv']

    # Check if any of the image extensions is in the URL
    return any(ext in url.lower() for ext in image_extensions)

async def crawl_reddit_subreddit(subreddit_name):

    reddit = define_reddit()

    subreddit = reddit.subreddit(subreddit_name)
    allPosts = []
    
    
    num_posts, post_time = count_posts_from_three_days_ago(subreddit_name)
    print("p time", post_time)
    print(f"Number of posts from yesterday in /r/{subreddit_name}: {num_posts}")
    
    rounded = math.ceil(num_posts / 100)
    #err with rounded 
    print("roundeed", rounded,"k328")
    num_posts = 10
    count1 = 1 
    #err here i think 
    for _ in range(rounded):
        print("Entering loop")
        
        count1 += 1
        
  
        print(num_posts)  
        count2 = 1
        # Now you can insert data into the 'newsarticle' table
        
        for index, submission in enumerate(subreddit.new(limit=num_posts)):
            try:
                print(count2,"count2 loop count   -=--0")
                count2 +=1
                print(index, "index / count ")
                print(submission.selftext)
                print(submission.title)
                
                print("for try")
                print(type(post_time))
                submission_time = datetime.utcfromtimestamp(submission.created_utc)
                if submission_time < post_time:
                  
                    break
            
                media = None 
                if submission.is_created_from_ads_ui:   
                     
                    continue
                if hasattr(submission, 'crosspost_parent'):
                
                    continue
                try:
                    
                    print(submission.media)
                finally:
                    pass    
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp']
                if submission.thumbnail == 'nsfw':
                    media = None
                  
                elif submission.media != None:
                    if submission.thumbnail is None:
                         if 'thumbnail_url' in submission.media:
                            media = submission.media['oembed']['thumbnail_url']
                            print("1")
                            # Now you have the 'thumbnail_url' value
                            
                         else:
                           
                            media = None
                            # 'thumbnail_url' doesn't exist in the 'media' dictionary
                            print("No thumbnail URL available.")
                    else: 
                      
                        media = submission.thumbnail
                     
                        print(type(submission.thumbnail))
                
                       
                        media = submission.thumbnail
                        print(media)
                

                elif submission.url != None and is_image_url(submission.url):
                   
                    
                    print("images")
                    print(submission.url)
                    urls = submission.url
                    print(type(urls))
                
                    media = submission.url
                else:
                    media = None
               
              
               
                        
                r = Rake()   
                sentiment = ""
                entitiesArray = {}
                
                print(submission)
                post_title = submission.title
                post_content = submission.selftext
                post_score = submission.score
                num_comments = submission.num_comments
                id = submission.id
                subreddit_name = submission.subreddit.display_name
                upVRatio  = submission.upvote_ratio
                ups = submission.ups
                unix_timestamp = submission.created
                creation_time = datetime.fromtimestamp(unix_timestamp)
                formatted_creation_time = creation_time.strftime('%Y-%m-%d %H:%M:%S')
                
            
                    
                print(creation_time)
                id = id[0]+id[1]+id[2]+id[3]+id[4]+id[5]+id[6]
                

                keywordsArray = []
             
                runT = False
                
                if post_content != None and len(post_content) > 0:
                    
                    print(len(post_content))
          
                    post_content = summarize(post_content)
                    
                 
                    post_content = remove_urls(post_content)
                    print(post_content)
                    print(len(post_content))
                    
                    if len(post_content) <= 0 or post_content == None:
                        runT = True
                    if runT != True:
                        
                        entitiesArray, sentiment  = get_entities(post_content)
                        keywordsArray = rake(post_content)  # Corrected line
                        
                        print("content - ", post_content)
                        print(type(post_content))
                else:
                    runT = True
                if runT:
                  
              
                    post_content = remove_urls(post_title)
                    
              
                    entitiesArray, sentiment = get_entities(post_title)
       
                    keywordsArray = rake(post_title)
             
                    print(type(post_title))
                letFormattedKeywords = []
                print("kwArray",keywordsArray)

                for item in keywordsArray:
                      print("start run")
                      letFormattedKeywords.append(item[1])
                    
                print(letFormattedKeywords, "xdontgiveittoya")
                print(type(letFormattedKeywords), "xdontgiveittoya")
                
                sent = 0
                if sentiment[0]['label']  == "negative":
                    sent -= 1
                elif sentiment[0]['label']  == "positive":
                    sent += 1
                elif sentiment[0]['label']  == "neutral":
                    pass
                else: 
                    print("err getting sent value")
                    
                posts_data = [id, post_title, post_content, post_score, num_comments,
                formatted_creation_time,  subreddit_name, entitiesArray, letFormattedKeywords, upVRatio, ups, media , sent ]
                print(posts_data)
                print(entitiesArray)
                print("kwArray",keywordsArray)
                
                allPosts.append(posts_data)
               
                print(index, "index / count ")
            except Exception as e:
                print(f"Error in loop iteration {index}: {str(e)}") 
            finally: 
                print("done")

    print(allPosts)
    print(len(allPosts))
    print("saveTODB________")
    saveToDB(allPosts)        
   

    # Function to crawl multiple subreddits concurrently with rate limiting and staggering
async def crawl_subreddits(subreddit_names):
    concurrency_limit = 3  # Adjust the concurrency limit as needed
    stagger_interval = 19  # Adjust the stagger interval as needed (in seconds)

    semaphores = asyncio.Semaphore(concurrency_limit)

    for subreddit_name in subreddit_names:
        async with semaphores:
            print(f"Crawling subreddit: {subreddit_name}")
            await crawl_reddit_subreddit(subreddit_name)
            await asyncio.sleep(stagger_interval)

async def main():
    #  list of subreddit names  
    subreddit_names = ["games","celebs","politics"]
    subreddit_namess = [ "Politics","economy", "games", "Business", "News", "Technology", "damnthatsinteresting", "Gadgets", "financialindependence", "Environment", "popculturechat", "Entertainment", "worldnews", "food", "nottheonion", "economics"]
 
    # Split the subreddit_names into three roughly equal parts
    chunk_size = len(subreddit_names) // 3
    chunks = [subreddit_names[i:i + chunk_size] for i in range(0, len(subreddit_names), chunk_size)]

    # Create tasks for each chunk                                                                                
    tasks = [asyncio.create_task(crawl_subreddits(chunk)) for chunk in chunks]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
    
    #crawl_get_values("games")
    

    
    # Function to save posts to the database
def saveToDB(allPosts):

    
    try:
        print("very start save")
        host = os.environ.get('crawler_host', 'localhost')  # Use environment variable or default to localhost)
        port =  os.environ.get("crawler_port")
        user = os.environ.get("crawler_user")
        password =  os.environ.get("crawler_pass")
        db_name =  os.environ.get("crawler_db")

        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            db=db_name,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

        formatted_allPosts = []  # Define the list outside the try block
        with connection.cursor() as cursor:
            print("connection start")
            insert_query = """
                INSERT INTO posts
                (id, post_title, post_content, post_score, num_comments, creation_time, 
                subreddit_name,  entitiesWithSent , keywordsArray, upVRatio, 
                ups, media, sent)
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            insert_totals_query = """
                INSERT INTO totals (entity, score, type, popularity)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE score = score + 1, popularity = popularity + VALUES(popularity);
                   
                    
            """
            print("for start ")
            print("zdontgiveittoya")
            for post in allPosts:
                print("for try ")
                try:
                   
                            
                    formatted_post = (
                        post[0],
                        post[1],
                        None if post[2] == "" else post[2],
                        post[3],
                        post[4],
                        post[5],
                        post[6],
                        json.dumps(post[7]) if post[7] is not None else None,
                        json.dumps(post[8]) if post[8] is not None else None,
                        post[9],
                        post[10],
                        None if post[11] == "" else post[11],
                        post[12],
                       

                    )
                    for x in post:
                        print(x)
                        print("type", type(x))
                    print(type(post[0]))
                    formatted_allPosts.append(formatted_post)
                    
                    entities_json = formatted_post[7]  # Get the JSON string
                    entities = json.loads(entities_json)  # Parse it into a Python data structure
                    print( "for start xxxxxx")
                    print(entities)
                    print(type(entities))
                    if entities is not None and len(entities) > 0 and len(entities[0]) == 2:
                        for entity in entities:
                            ent = entity[0]  # Access the first element from the sub-list
                            type1 = entity[1]  # Access the second element from the sub-list

                            
                            
                            print(post[12], "DDDD:")
                            # Assuming entities[2][i] refers to sentiment, you can add the logic here
                             

                     
                            # Assuming you have an insert query and a cursor, you can execute it like this:
                            totals_data = (ent, 1, type1, post[12])
                            print("insert totals query")
                            print(totals_data)
                            cursor.execute(insert_totals_query, totals_data)

                    

                    print("after")    
                    print(formatted_post[0]) 
                    
                    print(type(formatted_post[0] ))
                    cursor.execute(insert_query, formatted_post)
                    connection.commit()
                    

                except pymysql.err.IntegrityError as e:
                    if e.args[0] == 1062:
                        print("Duplicate entry. Skipping...")
                    else:
                        print("Error:", e)
                except Exception as e:
                    print("Error in inner loop:", e)

    except Exception as e:
        print("Error in outer loop:", e)
    finally:
        cursor.close()

    