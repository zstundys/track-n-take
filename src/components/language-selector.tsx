import React from "react";
import { useTranslation } from "react-i18next";
import { Check, ChevronsUpDown, Globe } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";

const languages = [
  { value: "en", label: "English" },
  { value: "lt", label: "Lietuvių" },
];

const LanguageSelector: React.FC = () => {
  const { i18n, t } = useTranslation();
  const [open, setOpen] = React.useState(false);

  const currentLanguage =
    languages.find((language) => language.value === i18n.language) ||
    languages[0];

  const handleLanguageChange = (value: string) => {
    i18n.changeLanguage(value);
    setOpen(false);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between"
          size="sm"
        >
          <div className="flex items-center gap-2">
            <Globe className="h-4 w-4" />
            <span>{currentLanguage.label}</span>
          </div>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[200px] p-0">
        <Command>
          <CommandEmpty>No language found.</CommandEmpty>
          <CommandList>
            <CommandGroup>
              {languages.map((language) => (
                <CommandItem
                  key={language.value}
                  value={language.value}
                  onSelect={() => handleLanguageChange(language.value)}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      i18n.language === language.value
                        ? "opacity-100"
                        : "opacity-0"
                    )}
                  />
                  {language.label}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default LanguageSelector;
