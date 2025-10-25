import { LucideIcon } from "lucide-react";
import { Card } from "@/components/ui/card";

interface StatsCardProps {
  icon: LucideIcon;
  label: string;
  value: string | number;
  description: string;
  trend?: "up" | "down";
  variant?: "default" | "success" | "warning" | "destructive";
}

export const StatsCard = ({
  icon: Icon,
  label,
  value,
  description,
  trend,
  variant = "default",
}: StatsCardProps) => {
  const variants = {
    default: "border-primary/20 bg-card hover:shadow-glow",
    success: "border-success/20 bg-success/5",
    warning: "border-warning/20 bg-warning/5",
    destructive: "border-destructive/20 bg-destructive/5",
  };

  const iconVariants = {
    default: "text-primary",
    success: "text-success",
    warning: "text-warning",
    destructive: "text-destructive",
  };

  return (
    <Card className={`p-6 transition-all duration-300 hover:scale-105 ${variants[variant]}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-muted-foreground mb-1">{label}</p>
          <div className="flex items-baseline gap-2">
            <h3 className="text-3xl font-bold text-foreground">{value}</h3>
            {trend && (
              <span className={`text-sm ${trend === "up" ? "text-success" : "text-destructive"}`}>
                {trend === "up" ? "↑" : "↓"}
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        </div>
        <div className={`p-3 rounded-xl bg-gradient-primary ${iconVariants[variant]}`}>
          <Icon className="h-6 w-6" />
        </div>
      </div>
    </Card>
  );
};
