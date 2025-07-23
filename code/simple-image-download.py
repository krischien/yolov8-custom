import simple_image_download.simple_image_download as simp

response = simp.Downloader()  # Correct way to instantiate the class

keywords = ["car"]

for kw in keywords:
    response.download(kw, 200)  # Downloads
