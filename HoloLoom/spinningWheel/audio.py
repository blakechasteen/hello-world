# In AudioSpinner class, update _enrich_shards method:

async def _enrich_shards(self, shards: List[MemoryShard]) -> List[MemoryShard]:
    """Enrich shards using modular enrichment strategies."""
    
    if not self.config.enable_enrichment:
        return shards
    
    # Initialize enrichers
    from .enrichment import MetadataEnricher, SemanticEnricher, TemporalEnricher
    
    metadata_enricher = MetadataEnricher()
    semantic_enricher = SemanticEnricher({'model': self.config.ollama_model})
    temporal_enricher = TemporalEnricher({'reference_date': '2025-10-13'})
    
    for shard in shards:
        enrichment = {}
        
        # Apply each enricher
        try:
            metadata_result = await metadata_enricher.enrich(shard.text)
            enrichment['metadata'] = metadata_result.data
        except Exception as e:
            enrichment['metadata_error'] = str(e)
        
        try:
            semantic_result = await semantic_enricher.enrich(shard.text)
            enrichment['semantic'] = semantic_result.data
            
            # Merge semantic entities into shard
            if semantic_result.data.get('entities'):
                shard.entities.extend(semantic_result.data['entities'])
                shard.entities = list(set(shard.entities))
            
            # Merge semantic motifs into shard
            if semantic_result.data.get('motifs'):
                shard.motifs.extend(semantic_result.data['motifs'])
                shard.motifs = list(set(shard.motifs))
        except Exception as e:
            enrichment['semantic_error'] = str(e)
        
        try:
            temporal_result = await temporal_enricher.enrich(shard.text)
            enrichment['temporal'] = temporal_result.data
        except Exception as e:
            enrichment['temporal_error'] = str(e)
        
        # Store all enrichment results in metadata
        if shard.metadata is None:
            shard.metadata = {}
        shard.metadata['enrichment'] = enrichment
    
    return shards