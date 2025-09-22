import asyncio
import os
import pandas as pd
from tqdm import tqdm
from groq import AsyncGroq  

# Set your Groq API key
os.environ["GROQ_API_KEY"] = ""
client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
groq_model = ""

# Parameters
MAX_CONCURRENT_REQUESTS = 3
RETRIES = 3
WAIT_TIME = 60  

async def ask_groq(prompt):
    """Send a prompt to Groq and handle rate limiting."""
    for attempt in range(RETRIES):
        try:
            response = await client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": "You are an assistant that answers based on bug report information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit_exceeded" in str(e) and attempt < RETRIES - 1:
                print(f"[Rate limit] Waiting {WAIT_TIME}s and retrying...")
                await asyncio.sleep(WAIT_TIME)
            else:
                print(f"Error during Groq API call: {e}")
                return ""
async def worker(queue, results):
    """Async worker to process prompts from the queue."""
    while True:
        index, prompt = await queue.get()
        answer = await ask_groq(prompt)
        results[index] = answer
        queue.task_done()

async def process_file(name):
    """Process one CSV file."""
    gold = pd.read_csv(f"{name}.csv")
    gold["IssueID"] = gold["IssueID"].astype(str)
    goldsetids = list(gold["IssueID"])

    base_filename, _ = os.path.splitext(name)

    for ids in tqdm(goldsetids, desc="Processing Issues", unit="file"):
        
        filename = ("" + name)
        file = pd.read_csv(filename)
        
        if not os.path.exists(filename):
            print(f"File {filename} not found, skipping...")
            continue  
        
        output_path = f"{base_filename}/{ids}.csv"
        if os.path.exists(output_path):
            print(f"File {output_path} already processed, skipping...")
            continue
        file = pd.read_csv(filename)[:10]
        dataframe = pd.DataFrame(file)
        dataframe["LlamaAnswer"] = ""

        queue = asyncio.Queue()
        results = {}

        # Prepare prompts and put them in the queue
        for index, row in dataframe.iterrows():
            question = row["Question"]
            question = row.Question
            context = row['Title'] + row['Description']
            prompt = f"""Answer Question on the bug report based on the relevant information.
            Here is a bug report which has incomplete information.
            ## Bug Report - \n{context}
            There is a follow-up question asking for missing information.
            Can you answer the question below based on the bug report?
            Here is the question.\n 
            ## Question- \n{question}\n ## Answer : \n """

            await queue.put((index, prompt))

        # Start worker tasks
        workers = [asyncio.create_task(worker(queue, results)) for _ in range(MAX_CONCURRENT_REQUESTS)]

        # Wait until the queue is fully processed
        await queue.join()

        # Cancel the workers
        for w in workers:
            w.cancel()

        # Update dataframe
        for index, answer in results.items():
            dataframe.at[index, "LlamaAnswer"] = answer if answer else ""

        # Save output
        os.makedirs(f"{base_filename}", exist_ok=True)
        dataframe.to_csv(output_path, index=False)

async def main():
    filelist = os.listdir("")
    for name in filelist:
        if name.lower() == 'readme.md':
            continue
        base_filename, _ = os.path.splitext(name)
        print(f"Starting with file------- {name}")
        await process_file(base_filename)
        print(f"Completed with file------- {name}")      

if __name__ == '__main__':
    asyncio.run(main())
