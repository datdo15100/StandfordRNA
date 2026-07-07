from dotenv import load_dotenv
import kagglehub

load_dotenv()

path = kagglehub.competition_download(
    "stanford-rna-3d-folding"
)

print(path)