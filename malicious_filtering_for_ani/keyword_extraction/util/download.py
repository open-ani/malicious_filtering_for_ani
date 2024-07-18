import json

import pandas as pd
import requests


def download_and_read_csv(url):
    """
    Downloads a CSV file from the given URL and reads it into a pandas DataFrame.

    Args:
    - url (str): The URL of the CSV file.

    Returns:
    - pd.DataFrame: A DataFrame containing the content of the CSV file.
    """
    # Download the content from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the download fails

    # Convert the content to a string
    csv_content = response.content.decode('utf-8')

    # Read the CSV content into a DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(csv_content))

    df.to_csv("hotel.csv", index=False)


if __name__ == "__main__":
    # Specify the URL of the JSON file
    url = 'https://cas.dandanplay.net/api/comment/175690008?from=0&related=b-ep713744-1,gm-32644&urls=https%3A%2F%2Fwww.bilibili.com%2Fbangumi%2Fplay%2Fep713744%3Ffrom_spmid%3D666.25.episode.0%7Chttps%3A%2F%2Fani.gamer.com.tw%2FanimeVideo.php%3Fsn%3D32644&layer=2&chConvert=0'

    # Make a GET request to fetch the exernal_raw JSON content
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON content
        json_data = response.json()

        # Save the JSON data to a file
        with open('dandanplaytest.json', 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4)
        print("JSON data has been downloaded and saved to data.json")
    else:
        print(f"Failed to download JSON data. HTTP Status code: {response.status_code}")
