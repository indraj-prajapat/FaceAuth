import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface MatchCardProps {
  match: any;
  compact?: boolean;
}

export const MatchCard = ({ match, compact = false }: MatchCardProps) => {
  const getSimilarityColor = (sim: number) => {
    if (sim >= 80) return "text-success";
    if (sim >= 60) return "text-warning";
    return "text-muted-foreground";
  };

  if (compact) {
    return (
      <div className="flex gap-3 p-3 bg-muted/30 rounded-lg">
        {match.db_image_url && (
          <img
            src={match.db_image_url}
            alt="Match"
            className="w-20 h-20 object-cover rounded-md border border-border"
          />
        )}
        <div className="flex-1 min-w-0">
          <p className="text-xs text-muted-foreground truncate">
            {match.db_id || "Unknown ID"}
          </p>
          <p className={`text-lg font-bold ${getSimilarityColor(match.avg_similarity || 0)}`}>
            {match.avg_similarity?.toFixed(1)}%
          </p>
          <div className="flex gap-1 mt-1">
            <Badge variant="outline" className="text-xs">
              A: {match.arcface_similarity?.toFixed(0)}
            </Badge>
            <Badge variant="outline" className="text-xs">
              E: {match.elastic_similarity?.toFixed(0)}
            </Badge>
          </div>
        </div>
      </div>
    );
  }

  return (
    <Card className="overflow-hidden hover:shadow-lg transition-all duration-300">
      {match.db_image_url && (
        <div className="aspect-square relative overflow-hidden bg-muted">
          <img
            src={match.db_image_url}
            alt="Database match"
            className="w-full h-full object-cover"
          />
          <div className="absolute top-2 right-2">
            <Badge className={getSimilarityColor(match.avg_similarity || 0)}>
              {match.avg_similarity?.toFixed(1)}%
            </Badge>
          </div>
        </div>
      )}
      
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">DB Index</span>
          <Badge variant="outline">{match.db_index}</Badge>
        </div>
        
        <div className="text-xs space-y-1">
          <p className="truncate text-muted-foreground">
            ID: {match.db_id || "N/A"}
          </p>
        </div>

        <div className="grid grid-cols-3 gap-1 text-xs">
          <div className="text-center p-1 bg-muted/50 rounded">
            <p className="text-muted-foreground">Arc</p>
            <p className="font-semibold">{match.arcface_similarity?.toFixed(0)}</p>
          </div>
          <div className="text-center p-1 bg-muted/50 rounded">
            <p className="text-muted-foreground">Ada</p>
            <p className="font-semibold">{match.adaface_similarity?.toFixed(0)}</p>
          </div>
          <div className="text-center p-1 bg-muted/50 rounded">
            <p className="text-muted-foreground">Ela</p>
            <p className="font-semibold">{match.elastic_similarity?.toFixed(0)}</p>
          </div>
        </div>
      </div>
    </Card>
  );
};
