from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BoundingBox(BaseModel):
    west: float = Field(..., description="Западная долгота")
    south: float = Field(..., description="Южная широта")
    east: float = Field(..., description="Восточная долгота")
    north: float = Field(..., description="Северная широта")

class LogisticsPoint(BaseModel):
    lat: float
    lon: float
    tags: Dict[str, Any]

class MSTEdge(BaseModel):
    from_index: int
    to_index: int
    distance: float

class MSTResponse(BaseModel):
    nodes_count: int
    edges_count: int
    total_distance: float
    points: List[LogisticsPoint]
    edges: List[MSTEdge]

class MapResponse(BaseModel):
    html_content: str
    points_count: int
    edges_count: int

class MSTAnalysisResponse(BaseModel):
    status: str
    metric_used: str
    analysis_map: str
    metrics: Dict[str, Dict[int, float]]