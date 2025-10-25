import { useState, useEffect } from "react";
import { AlertCircle, CheckCircle, XCircle, AlertTriangle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { MatchCard } from "@/components/MatchCard";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

interface ReviewSectionProps {
  onUpdate?: () => void;
}

export const ReviewSection = ({ onUpdate }: ReviewSectionProps) => {
  const [queue, setQueue] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("all");
  const [reviewNotes, setReviewNotes] = useState<{ [key: string]: string }>({});
  const { toast } = useToast();

  const fetchQueue = async () => {
    try {
      const url = new URL(`${API_BASE}/api/review/queue`);
      if (filter !== "all") url.searchParams.set("risk_level", filter.toUpperCase());

      const res = await fetch(url.toString());
      if (!res.ok) throw new Error("Failed to fetch review queue");
      
      const data = await res.json();
      setQueue(data);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to load review queue",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQueue();
  }, [filter]);

  const handleDecision = async (probeId: string, decision: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/review/decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          probe_id: probeId,
          decision,
          reviewer_notes: reviewNotes[probeId] || "",
        }),
      });

      if (!res.ok) throw new Error("Failed to submit decision");

      toast({
        title: "Decision Submitted",
        description: `${decision} decision recorded`,
      });

      setReviewNotes((prev) => {
        const updated = { ...prev };
        delete updated[probeId];
        return updated;
      });

      fetchQueue();
      onUpdate?.();
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to submit decision",
        variant: "destructive",
      });
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level?.toUpperCase()) {
      case "HIGH":
        return <XCircle className="h-5 w-5 text-destructive" />;
      case "MEDIUM":
        return <AlertTriangle className="h-5 w-5 text-warning" />;
      case "LOW":
        return <CheckCircle className="h-5 w-5 text-success" />;
      default:
        return <AlertCircle className="h-5 w-5 text-muted-foreground" />;
    }
  };

  const getRiskBadgeVariant = (level: string): "destructive" | "default" | "secondary" => {
    switch (level?.toUpperCase()) {
      case "HIGH":
        return "destructive";
      case "MEDIUM":
        return "default";
      default:
        return "secondary";
    }
  };

  if (loading) {
    return (
      <Card className="p-12 text-center">
        <p className="text-muted-foreground">Loading review queue...</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-semibold">Manual Review Queue</h3>
        <Select value={filter} onValueChange={setFilter}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by risk" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Risks</SelectItem>
            <SelectItem value="high">High Risk</SelectItem>
            <SelectItem value="medium">Medium Risk</SelectItem>
            <SelectItem value="low">Low Risk</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {queue.length === 0 ? (
        <Card className="p-12 text-center">
          <CheckCircle className="h-16 w-16 text-success mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">All Clear!</h3>
          <p className="text-muted-foreground">No items in the review queue</p>
        </Card>
      ) : (
        <div className="space-y-6">
          {queue.map((item) => (
            <Card key={item.id} className="p-6">
              <div className="grid lg:grid-cols-[300px_1fr_300px] gap-6">
                {/* Probe Images */}
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    {getRiskIcon(item.risk_level)}
                    <h4 className="font-semibold">Probe Images</h4>
                  </div>
                  
                  <div className="space-y-3">
                    {item.probe_images?.aligned && (
                      <div className="relative group">
                        <img
                          src={item.probe_images.aligned}
                          alt="Aligned"
                          className="w-full rounded-lg border-2 border-primary/20"
                        />
                        <p className="text-xs text-center mt-1 text-muted-foreground">Aligned</p>
                      </div>
                    )}
                    {item.probe_images?.enhanced && (
                      <div className="relative group">
                        <img
                          src={item.probe_images.enhanced}
                          alt="Enhanced"
                          className="w-full rounded-lg border-2 border-secondary/20"
                        />
                        <p className="text-xs text-center mt-1 text-muted-foreground">Enhanced</p>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 space-y-2">
                    <Badge variant={getRiskBadgeVariant(item.risk_level)}>
                      {item.risk_level} RISK
                    </Badge>
                    <p className="text-xs text-muted-foreground">
                      Probe ID: {item.probe_id}
                    </p>
                  </div>
                </div>

                {/* Matches and Details */}
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-3">Potential Matches</h4>
                    {item.matches && item.matches.length > 0 ? (
                      <div className="grid grid-cols-2 gap-4">
                        {item.matches.slice(0, 4).map((match: any, idx: number) => (
                          <MatchCard key={idx} match={match} />
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">No matches found</p>
                    )}
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">Analysis Summary</h4>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {item.summary && Object.entries(item.summary).map(([key, value]: [string, any]) => (
                        <div key={key} className="bg-muted/50 rounded-lg p-3">
                          <p className="text-xs text-muted-foreground">{key}</p>
                          <p className="text-sm font-semibold">{value}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {item.why_review && (
                    <div className="bg-warning/10 border border-warning/20 rounded-lg p-3">
                      <p className="text-sm">
                        <span className="font-semibold">Reason: </span>
                        {item.why_review}
                      </p>
                    </div>
                  )}
                </div>

                {/* Decision Panel */}
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-2">Review Notes</h4>
                    <Textarea
                      placeholder="Add your review notes..."
                      value={reviewNotes[item.probe_id] || ""}
                      onChange={(e) =>
                        setReviewNotes((prev) => ({
                          ...prev,
                          [item.probe_id]: e.target.value,
                        }))
                      }
                      rows={4}
                    />
                  </div>

                  <div className="space-y-2">
                    <Button
                      onClick={() => handleDecision(item.probe_id, "approve")}
                      className="w-full bg-success hover:bg-success/90"
                    >
                      <CheckCircle className="mr-2 h-4 w-4" />
                      Approve
                    </Button>
                    <Button
                      onClick={() => handleDecision(item.probe_id, "reject")}
                      variant="destructive"
                      className="w-full"
                    >
                      <XCircle className="mr-2 h-4 w-4" />
                      Reject
                    </Button>
                    <Button
                      onClick={() => handleDecision(item.probe_id, "escalate")}
                      variant="outline"
                      className="w-full"
                    >
                      <AlertTriangle className="mr-2 h-4 w-4" />
                      Escalate
                    </Button>
                  </div>

                  <div className="pt-4 border-t text-xs text-muted-foreground space-y-1">
                    <p>Status: {item.status}</p>
                    <p>Branch: {item.branch}</p>
                    <p>Created: {new Date(item.created_at).toLocaleString()}</p>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};
