import React, { useState } from "react";
import { format, addDays } from "date-fns";
import { Calendar as CalendarIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { useIntl } from "@/hooks/useIntl";

interface DatePickerWithPresetsProps {
  value: Date | undefined;
  onDateChange: (date: Date | undefined) => void;
  placeholder: string;
  label?: string;
  fromDate?: Date;
  toDate?: Date;
  className?: string;
  presets?: { days: number; label: string }[];
}

const DEFAULT_PRESETS = [
  { days: 2, label: "+2d" },
  { days: 7, label: "+7d" },
  { days: 14, label: "+14d" },
  { days: 30, label: "+30d" },
];

export const DatePickerWithPresets: React.FC<DatePickerWithPresetsProps> = ({
  value,
  onDateChange,
  placeholder,
  fromDate,
  toDate,
  presets = DEFAULT_PRESETS,
  className,
}) => {
  const [isCalendarOpen, setIsCalendarOpen] = useState(false);
  const { shortDate } = useIntl();

  const handleSetPresetDate = (daysToAdd: number) => {
    const newDate = addDays(new Date(), daysToAdd);
    onDateChange(newDate);
  };

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      <Popover open={isCalendarOpen} onOpenChange={setIsCalendarOpen} modal>
        <PopoverTrigger asChild>
          <div className="flex flex-col gap-2  sm:w-[280px]">
            <Button
              type="button"
              variant={"outline"}
              className={cn(
                "w-full justify-start text-left font-normal",
                !value && "text-muted-foreground"
              )}
            >
              <CalendarIcon className="mr-2 h-4 w-4" />
              {value ? shortDate.format(value) : placeholder}
            </Button>
            {presets.length > 0 && (
              <div className="flex gap-2 min-w-0">
                {presets.map((preset, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    size="sm"
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSetPresetDate(preset.days);
                    }}
                    className="flex-1 min-w-[60px]"
                  >
                    {preset.label}
                  </Button>
                ))}
              </div>
            )}
          </div>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0">
          <Calendar
            mode="single"
            selected={value}
            onSelect={(date) => {
              setIsCalendarOpen(false);
              onDateChange(date);
            }}
            fromDate={fromDate}
            toDate={toDate}
            initialFocus
          />
        </PopoverContent>
      </Popover>
    </div>
  );
};

export default DatePickerWithPresets;
