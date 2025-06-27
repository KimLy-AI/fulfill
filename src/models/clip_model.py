import pandas as pd
import ast
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import pathlib
from .load_models import load_clip_model
from src.utils.image_utils import load_and_preprocess_image
from src.data_uploading.create_embedding import create_embeddings
import os
import torch
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

class CLIPSimilaritySearcher:
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        self.model, self.preprocess, self.device = load_clip_model(model_name, device)
        self.embedd_df = None
        self.dataset_features = None
        self.create_embedding = create_embeddings(dataset_folder='design_cropped')
    
    def load_embeddings(self, embedding_file: str = 'embedding_vector_test.csv') -> None:
        try:
            if os.getenv('CREATE_EMBEDDING') == True:
                self.create_embedding
            project_base_dir = pathlib.Path(__file__).parent.parent.resolve() 
            input_path = project_base_dir.joinpath(f'data/database/{embedding_file}')
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Embedding file not found: {input_path}")
            
            self.embedd_df = pd.read_csv(input_path, converters={'embedded': ast.literal_eval})
            embeddings_list = self.embedd_df['embedded'].tolist()
            self.dataset_features = torch.tensor(embeddings_list, dtype=torch.float32).to(self.device)
            self.dataset_features = self.dataset_features / self.dataset_features.norm(dim=-1, keepdim=True)
            logger.info(f"Loaded {len(self.embedd_df)} embeddings from {input_path}")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def encode_image(self, image_path: str) -> torch.Tensor:
        try:
            image_input = load_and_preprocess_image(image_path, self.preprocess)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input.unsqueeze(0).to(self.device))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def calculate_similarities(self, query_features: torch.Tensor) -> np.ndarray:
        if self.dataset_features is None:
            raise ValueError("Dataset features not loaded. Call load_embeddings() first.")
        similarity_scores = (query_features @ self.dataset_features.T).squeeze(0)
        return similarity_scores.cpu().numpy()
    
    def get_top_k_similar(self, similarity_scores: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        top_indices = np.argsort(similarity_scores)[::-1][:k]
        results = [(self.embedd_df.iloc[idx]['img_path'], float(similarity_scores[idx])) for idx in top_indices]
        return results
    
    def extract_image_id(self, image_path: str) -> str:
        return Path(image_path).stem.split('_')[-1].strip()
    
    def analyze_ranking(self, query_path: str, top_results: List[Tuple[str, float]]) -> Dict:
        query_id = self.extract_image_id(query_path)
        result = {
            'query_path': query_path, 'query_id': query_id, 'found_in_top_1': False,
            'found_in_top_3': False, 'found_in_top_5': False, 'found_in_top_10': False,
            'rank': None, 'similarity_score': None
        }
        for rank, (path, score) in enumerate(top_results, 1):
            if query_id in path:
                result.update({'rank': rank, 'similarity_score': score, 'found_in_top_1': rank <= 1,
                               'found_in_top_3': rank <= 3, 'found_in_top_5': rank <= 5, 'found_in_top_10': rank <= 10})
                break
        return result
    
    def process_single_image(self, image_path: str, top_k: int = 10) -> Dict:
        try:
            query_features = self.encode_image(image_path)
            similarity_scores = self.calculate_similarities(query_features)
            top_results = self.get_top_k_similar(similarity_scores, top_k)
            analysis = self.analyze_ranking(image_path, top_results)
            analysis['top_results'] = top_results
            return analysis
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {'error': str(e), 'query_path': image_path}
    
    def process_directory(self, input_dir: str, top_k: int = 10) -> List[Dict]:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if Path(f).suffix.lower() in image_extensions]
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return []
        logger.info(f"Processing {len(image_files)} images from {input_dir}")
        results = [self.process_single_image(image_path, top_k) for image_path in image_files]
        for result in results:
            if 'error' not in result:
                self._log_result(result)
        return results
    
    def _log_result(self, result: Dict) -> None:
        query_id = result['query_id']
        if result['found_in_top_1']:
            logger.info(f"✓ {query_id}: Found in TOP 1 (score: {result['similarity_score']:.4f})")
        elif result['found_in_top_3']:
            logger.info(f"✓ {query_id}: Found in TOP 3 (rank: {result['rank']}, score: {result['similarity_score']:.4f})")
        elif result['found_in_top_5']:
            logger.info(f"○ {query_id}: Found in TOP 5 (rank: {result['rank']}, score: {result['similarity_score']:.4f})")
        elif result['found_in_top_10']:
            logger.info(f"○ {query_id}: Found in TOP 10 (rank: {result['rank']}, score: {result['similarity_score']:.4f})")
        else:
            logger.info(f"✗ {query_id}: Not found in TOP 10")
    
    def get_summary_stats(self, results: List[Dict]) -> Dict:
        valid_results = [r for r in results if 'error' not in r]
        total = len(valid_results)
        if total == 0:
            return {'total': 0, 'error': 'No valid results'}
        stats = {
            'total': total, 'top_1': sum(1 for r in valid_results if r['found_in_top_1']),
            'top_3': sum(1 for r in valid_results if r['found_in_top_3']),
            'top_5': sum(1 for r in valid_results if r['found_in_top_5']),
            'top_10': sum(1 for r in valid_results if r['found_in_top_10']),
        }
        for key in ['top_1', 'top_3', 'top_5', 'top_10']:
            stats[f'{key}_pct'] = (stats[key] / total) * 100
        return stats
    
    def save_results_to_csv(self, results: List[Dict], output_file: str = None) -> str:
        script_location = pathlib.Path(__file__).parent.parent.resolve()
        output_directory = script_location.joinpath("data/result/csv_file/")
        output_directory.mkdir(parents=True, exist_ok=True)
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"similarity_search_results_{timestamp}.csv"
        output_path = output_directory / output_file
        csv_data = []
        for result in results:
            if 'error' in result:
                csv_data.append({'query_path': result['query_path'], 'query_id': 'ERROR', 'rank': None,
                                 'similarity_score': None, 'found_in_top_1': False, 'found_in_top_3': False,
                                 'found_in_top_5': False, 'found_in_top_10': False, 'error': result['error']})
            else:
                row = {'query_path': result['query_path'], 'query_id': result['query_id'], 'rank': result['rank'],
                       'similarity_score': result['similarity_score'], 'found_in_top_1': result['found_in_top_1'],
                       'found_in_top_3': result['found_in_top_3'], 'found_in_top_5': result['found_in_top_5'],
                       'found_in_top_10': result['found_in_top_10'], 'error': None}
                for i, (path, score) in enumerate(result['top_results'], 1):
                    row[f'top_{i}_path'] = path
                    row[f'top_{i}_score'] = score
                csv_data.append(row)
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)

    def main():
        CLIPSimilaritySearcher = CLIPSimilaritySearcher()
        CLIPSimilaritySearcher.load_embeddings
if __name__ == '__main__':
    pass