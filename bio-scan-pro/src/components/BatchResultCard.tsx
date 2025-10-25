import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, CheckCircle, Shield } from "lucide-react";
import { MatchCard } from "@/components/MatchCard";

interface BatchResultCardProps {
  filename: string;
  result: any;
}

export const BatchResultCard = ({ filename, result }: BatchResultCardProps) => {
  const getDecisionIcon = () => {
    switch (result.decision) {
      case "unique":
      case "accept":
        return <CheckCircle className="h-5 w-5 text-success" />;
      case "reject":
        return <AlertCircle className="h-5 w-5 text-destructive" />;
      default:
        return <Shield className="h-5 w-5 text-warning" />;
    }
  };

  const getRiskBadgeVariant = (): "destructive" | "default" | "secondary" => {
    const level = result.risk_level?.toUpperCase();
    switch (level) {
      case "HIGH":
        return "destructive";
      case "MEDIUM":
        return "default";
      default:
        return "secondary";
    }
  };

  const topMatches = result.matches?.slice(0, 3) || [];

  return (
    <Card className="p-6 animate-scale-in">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          {getDecisionIcon()}
          <div>
            <h4 className="font-semibold">{filename}</h4>
            <p className="text-sm text-muted-foreground">{result.decision}</p>
          </div>
        </div>
        <Badge variant={getRiskBadgeVariant()}>
          {result.risk_level}
        </Badge>
      </div>

      <div className="grid md:grid-cols-[240px_1fr] gap-6">
        {/* Probe Images */}
        <div className="space-y-3">
          {result.probe?.image_url_orig && (
            <div>
              <img
                src={result.probe.image_url_orig}
                alt="Original"
                className="w-full rounded-lg border-2 border-primary/20"
              />
              <p className="text-xs text-center mt-1 text-muted-foreground">Original</p>
            </div>
          )}
          {result.probe?.image_url_enh && (
            <div>
              <img
                src={result.probe.image_url_enh}
                alt="Enhanced"
                className="w-full rounded-lg border-2 border-secondary/20"
              />
              <p className="text-xs text-center mt-1 text-muted-foreground">Enhanced</p>
            </div>
          )}
        </div>

        {/* Details and Matches */}
        <div className="space-y-4">
          {/* Metrics */}
          <div>
            <h5 className="text-sm font-semibold mb-2">Metrics</h5>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <div className="bg-muted/50 rounded-lg p-2 text-center">
                <p className="text-xs text-muted-foreground">Risk Score</p>
                <p className="text-sm font-bold">{result.risk_score?.toFixed(2)}</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-2 text-center">
                <p className="text-xs text-muted-foreground">Quality</p>
                <p className="text-sm font-bold">{result.quality_score?.toFixed(2)}</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-2 text-center">
                <p className="text-xs text-muted-foreground">Similarity</p>
                <p className="text-sm font-bold">{result.signals?.similarity?.toFixed(2)}</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-2 text-center">
                <p className="text-xs text-muted-foreground">Morph</p>
                <p className="text-sm font-bold">{result.signals?.morph?.toFixed(2)}</p>
              </div>
            </div>
          </div>

          {/* Matches */}
          {topMatches.length > 0 && (
            <div>
              <h5 className="text-sm font-semibold mb-2">
                Top Matches ({topMatches.length})
              </h5>
              <div className="grid grid-cols-3 gap-3">
                {topMatches.map((match: any, idx: number) => (
                  <MatchCard key={idx} match={match} />
                ))}
              </div>
            </div>
          )}

          {/* Best Match Info */}
          {result.best_match && (
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
              <p className="text-xs font-semibold mb-1">Best Match</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">
                  {result.best_match.db_id || "Unknown"}
                </span>
                <span className="font-bold text-primary">
                  {result.best_match.similarity}% similar
                </span>
              </div>
            </div>
          )}

          {result.reason && (
            <div className="text-xs text-muted-foreground italic">
              {result.reason}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};
