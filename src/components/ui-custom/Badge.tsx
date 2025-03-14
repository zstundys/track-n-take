
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground hover:bg-primary/80",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/80",
        outline:
          "text-foreground border border-input hover:bg-accent hover:text-accent-foreground",
        green: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-100",
        blue: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-100",
        red: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-100",
        yellow: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-100",
        purple: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-100",
        gray: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100",
        orange: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-100",
        cyan: "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-100",
      },
      size: {
        default: "h-6 px-2.5 py-0.5 text-xs",
        sm: "h-5 px-2 py-0 text-xs",
        lg: "h-7 px-3 py-1 text-sm",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {
  animated?: boolean;
}

function Badge({ 
  className, 
  variant, 
  size, 
  animated = false,
  ...props 
}: BadgeProps) {
  const Component = animated ? motion.div : "div";
  const animationProps = animated ? {
    initial: { scale: 0.8, opacity: 0 },
    animate: { scale: 1, opacity: 1 },
    transition: { duration: 0.2 }
  } : {};

  return (
    <Component
      className={cn(badgeVariants({ variant, size }), className)}
      {...animationProps}
      {...props}
    />
  );
}

export { Badge, badgeVariants };
