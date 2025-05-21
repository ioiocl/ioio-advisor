import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

class LocalStorageService:
    """Local file system storage service for saving generated images."""
    
    def __init__(self, base_dir: str = None):
        """Initialize the storage service with a base directory."""
        if base_dir is None:
            # Get the project root directory (where the main application file is)
            project_root = Path(__file__).parent.parent.parent.parent
            self.base_dir = project_root / 'images'
        else:
            self.base_dir = Path(base_dir)
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """Create the storage directory if it doesn't exist."""
        os.makedirs(self.base_dir, exist_ok=True)
    
    def _generate_filename(self, extension: str = "jpeg") -> str:
        """Generate a unique filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}.{extension}"
    
    async def save_image(self, image_data: bytes, extension: str = "jpeg") -> str:
        """
        Save image data to the file system.
        
        Args:
            image_data: The binary image data to save
            extension: The file extension (default: png)
            
        Returns:
            str: The URL path to the saved image
        """
        filename = self._generate_filename(extension)
        file_path = self.base_dir / filename
        
        try:
            # Validate that the image data is not empty
            if not image_data:
                raise ValueError("Empty image data")
                
            # Save the image data
            with open(file_path, "wb") as f:
                f.write(image_data)
                
            # Verify the file was written successfully
            if not file_path.exists() or file_path.stat().st_size == 0:
                raise ValueError("Failed to write image file")
        except Exception as e:
            # Clean up any partially written file
            if file_path.exists():
                file_path.unlink()
            raise e
        
        # Return the URL path
        return f"/images/{filename}"
    
    def get_image_url(self, path: str) -> str:
        """
        Convert a storage path to a full URL.
        
        Args:
            path: The storage path returned by save_image
            
        Returns:
            str: The full URL to access the image
        """
        return f"/static/{path}"
