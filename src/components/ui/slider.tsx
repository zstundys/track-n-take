import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";

import { cn } from "@/lib/utils";
import { Button } from "./button";
import { Minus, Plus } from "lucide-react";
import { Input } from "./input";

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => {
  const value = props.value?.[0];
  const min = props.min ?? 0;
  const max = props.max ?? 100;

  return (
    <div className="flex items-center gap-4">
      <Button
        type="button"
        variant="outline"
        size="icon"
        onClick={() =>
          props.onValueChange?.([Math.max((value ?? min) - 1, min)])
        }
        disabled={props.disabled}
      >
        <Minus className="h-4 w-4" />
      </Button>
      <div className="flex-1">
        <SliderPrimitive.Root
          ref={ref}
          className={cn(
            "relative flex w-full h-10 touch-none select-none items-center",
            className
          )}
          {...props}
        >
          <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary">
            <SliderPrimitive.Range className="absolute h-full bg-primary" />
          </SliderPrimitive.Track>
          <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50" />
        </SliderPrimitive.Root>
      </div>
      <Button
        type="button"
        variant="outline"
        size="icon"
        onClick={() =>
          props.onValueChange?.([Math.min((value ?? min) + 1, max)])
        }
      >
        <Plus className="h-4 w-4" />
      </Button>
      <div className="w-16">
        <Input
          type="number"
          min={min}
          max={max}
          value={value}
          onChange={(e) => props.onValueChange?.([Number(e.target.value)])}
        />
      </div>
    </div>
  );
});
Slider.displayName = SliderPrimitive.Root.displayName;

export { Slider };
