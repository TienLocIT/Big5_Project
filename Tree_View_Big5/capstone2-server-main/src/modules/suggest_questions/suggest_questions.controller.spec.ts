import { Test, TestingModule } from '@nestjs/testing';
import { SuggestQuestionsController } from './suggest_questions.controller';
import { SuggestQuestionsService } from './suggest_questions.service';

describe('SuggestQuestionsController', () => {
  let controller: SuggestQuestionsController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [SuggestQuestionsController],
      providers: [SuggestQuestionsService],
    }).compile();

    controller = module.get<SuggestQuestionsController>(SuggestQuestionsController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
