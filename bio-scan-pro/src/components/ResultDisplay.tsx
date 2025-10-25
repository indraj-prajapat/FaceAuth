import { AlertCircle, CheckCircle, Shield, TrendingUp } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { MatchCard } from "@/components/MatchCard";

interface ResultDisplayProps {
  result: any;
}

export const ResultDisplay = ({ result }: ResultDisplayProps) => {
  const getDecisionIcon = () => {
    switch (result.decision) {
      case "unique":
      case "accept":
        return <CheckCircle className="h-8 w-8 text-success" />;
      case "reject":
        return <AlertCircle className="h-8 w-8 text-destructive" />;
      default:
        return <Shield className="h-8 w-8 text-warning" />;
    }
  };

  const getDecisionColor = () => {
    switch (result.decision) {
      case "unique":
      case "accept":
        return "text-success";
      case "reject":
        return "text-destructive";
      default:
        return "text-warning";
    }
  };

  const getRiskColor = () => {
    const level = result.risk_level?.toUpperCase();
    switch (level) {
      case "LOW":
        return "bg-success/10 text-success border-success/20";
      case "MEDIUM":
        return "bg-warning/10 text-warning border-warning/20";
      case "HIGH":
        return "bg-destructive/10 text-destructive border-destructive/20";
      default:
        return "bg-muted/10 text-muted-foreground border-muted/20";
    }
  };

  const topMatches = result.matches?.slice(0, 3) || [];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Decision Header */}
      <Card className="p-6 bg-gradient-card">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-background rounded-full">
            {getDecisionIcon()}
          </div>
          <div className="flex-1">
            <h3 className={`text-2xl font-bold ${getDecisionColor()}`}>
              {result.decision?.toUpperCase()}
            </h3>
            <p className="text-muted-foreground">{result.reason || result.action}</p>
          </div>
          <div className={`px-4 py-2 rounded-lg border-2 ${getRiskColor()}`}>
            <p className="text-sm font-semibold">{result.risk_level} RISK</p>
            <p className="text-xs opacity-80">Score: {result.risk_score?.toFixed(2)}</p>
          </div>
        </div>
      </Card>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Probe Images */}
        <Card className="p-6">
          <h4 className="font-semibold mb-4 flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Probe Images
          </h4>
          
          <div className="space-y-4">
            {result.probe?.image_url_orig && (
              <div>
                <img
                  src={result.probe.image_url_orig}
                  alt="Original"
                  className="w-full rounded-lg border-2 border-primary/20"
                />
                <p className="text-xs text-center mt-2 text-muted-foreground">Original</p>
              </div>
            )}
            
            {result.probe?.image_url_enh && (
              <div>
                <img
                  src={result.probe.image_url_enh}
                  alt="Enhanced"
                  className="w-full rounded-lg border-2 border-secondary/20"
                />
                <p className="text-xs text-center mt-2 text-muted-foreground">Enhanced</p>
              </div>
            )}

            <div className="pt-4 space-y-2 border-t">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Quality Score</span>
                <span className="font-semibold">{result.quality_score?.toFixed(2)}%</span>
              </div>
              <Progress value={result.quality_score || 0} className="h-2" />
              
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Branch Used</span>
                <Badge variant="outline">{result.branch}</Badge>
              </div>
            </div>
          </div>
        </Card>

        {/* Analysis Signals */}
        <Card className="p-6">
          <h4 className="font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Analysis Signals
          </h4>
          
          <div className="space-y-4">
            {result.signals && Object.entries(result.signals).map(([key, value]: [string, any]) => {
              const percentage = typeof value === 'number' ? value : 0;
              return (
                <div key={key}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="capitalize text-muted-foreground">{key}</span>
                    <span className="font-semibold">{percentage.toFixed(2)}</span>
                  </div>
                  <Progress value={Math.abs(percentage)} className="h-2" />
                </div>
              );
            })}
          </div>

          {result.best_match && (
            <div className="mt-6 pt-6 border-t">
              <h5 className="text-sm font-semibold mb-3">Best Match</h5>
              <div className="relative group">
                {result.best_match.db_image_url && (
                  <img
                    src={result.best_match.db_image_url}
                    alt="Best match"
                    className="w-full rounded-lg border-2 border-primary/20"
                  />
                )}
                <div className="mt-2 text-xs space-y-1">
                  <p className="text-muted-foreground">
                    DB ID: {result.best_match.db_id || "N/A"}
                  </p>
                  <p className="font-semibold text-primary">
                    Similarity: {result.best_match.similarity}%
                  </p>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Top Matches */}
        <Card className="p-6">
          <h4 className="font-semibold mb-4">
            Top Matches {topMatches.length > 0 && `(${topMatches.length})`}
          </h4>
          
          {topMatches.length > 0 ? (
            <div className="space-y-4">
              {topMatches.map((match: any, idx: number) => (
                <MatchCard key={idx} match={match} compact />
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <CheckCircle className="h-12 w-12 text-muted-foreground mx-auto mb-2 opacity-50" />
              <p className="text-sm text-muted-foreground">No matches found</p>
              <p className="text-xs text-muted-foreground mt-1">This face is unique in the database</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};
